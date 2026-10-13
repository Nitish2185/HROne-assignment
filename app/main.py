"""
Employee Attendance & Analytics API - STARTER

Run:  uvicorn app.main:app --port 8000
Env:  MONGO_URI, MONGO_DB (a local .env is loaded for convenience)

This file was written quickly by a colleague who has left the company. The happy path
works, but nobody has reviewed it. Read PROBLEM_STATEMENT.docx for what is expected of you,
openapi.yaml for the contract and DATA_MODEL.md for what is stored in MongoDB.
"""
import logging
import os
import re
from contextlib import asynccontextmanager
from datetime import date, datetime, time, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal
from typing import Annotated, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, Field, model_validator
from pymongo import ASCENDING, DESCENDING, IndexModel, MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

# override=False (the default): real environment variables win over the .env file.
load_dotenv(override=False)

# tz_aware=True: PyMongo hands back timezone-aware UTC datetimes instead of naive ones.
client = MongoClient(
    os.getenv("MONGO_URI", "mongodb://localhost:27017"),
    tz_aware=True,
    tzinfo=timezone.utc,
    serverSelectionTimeoutMS=5000,
)
db = client[os.getenv("MONGO_DB", "attendance_db")]

IST = timezone(timedelta(hours=5, minutes=30))
log = logging.getLogger("attendance")


# --------------------------------------------------------------------------- #
# Indexes (created at startup, idempotently - create_indexes is a no-op when an
# identical index already exists)
# --------------------------------------------------------------------------- #
EMPLOYEE_INDEXES = [
    # emp_code is the business key: uniqueness is what makes concurrent creates safe.
    IndexModel([("emp_code", ASCENDING)], unique=True, name="uniq_emp_code"),
    # GET /employees?department=..., sorted by emp_code; also the trend's $match.
    IndexModel([("department", ASCENDING), ("emp_code", ASCENDING)], name="department_emp_code"),
    # Department summary: employees who joined on or before the end of the month.
    IndexModel([("joined_on", ASCENDING)], name="joined_on"),
]

ATTENDANCE_INDEXES = [
    # One record per (emp_code, date): makes punch-in race-safe; also serves every
    # per-employee lookup ($lookup in analytics, /attendance?emp_code=...).
    IndexModel([("emp_code", ASCENDING), ("date", ASCENDING)], unique=True, name="uniq_emp_code_date"),
    # GET /attendance default sort (date desc, emp_code asc) and month-range analytics.
    IndexModel([("date", DESCENDING), ("emp_code", ASCENDING)], name="date_desc_emp_code"),
    # GET /attendance?status=... (equality, then sort).
    IndexModel([("status", ASCENDING), ("date", DESCENDING), ("emp_code", ASCENDING)], name="status_date_emp_code"),
    # Punch-out: the employee's latest record with punch_in <= punched_at.
    IndexModel([("emp_code", ASCENDING), ("punch_in", DESCENDING)], name="emp_code_punch_in"),
]


def ensure_indexes() -> None:
    db.employees.create_indexes(EMPLOYEE_INDEXES)
    db.attendance_logs.create_indexes(ATTENDANCE_INDEXES)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    try:
        ensure_indexes()
    except PyMongoError as exc:  # keep starting; /health reports 503 until MongoDB answers
        log.error("could not create indexes at startup: %s", exc)
    yield


app = FastAPI(title="Employee Attendance & Analytics API", version="2.0.0", lifespan=lifespan)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
MIN_EPOCH_MS = 100_000_000_000  # anything smaller looks like seconds
MAX_EPOCH_MS = 4_102_444_800_000  # 2100-01-01
GRACE_SECONDS = 10 * 60
MIN_OVERTIME_MINUTES = 30
HALF_DAY_BELOW_HOURS = Decimal("4.50")
MAX_SHIFT = timedelta(hours=24)

PRESENCE_STATUSES = ("PRESENT", "WFH", "ON_DUTY")
ABSENCE_STATUSES = ("ABSENT", "LEAVE")

# Strict: rejects floats (1.5 and 1.0), numeric strings and booleans with a 422.
EpochMs = Annotated[int, Field(strict=True, ge=MIN_EPOCH_MS, le=MAX_EPOCH_MS)]
Status = Literal["PRESENT", "ABSENT", "LEAVE", "WFH", "ON_DUTY"]
PresenceStatus = Literal["PRESENT", "WFH", "ON_DUTY"]


def as_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Normalise a datetime read from MongoDB; a naive one is UTC by BSON definition."""
    if dt is None:
        return None
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def ms_to_dt(ms: int) -> datetime:
    """Epoch ms -> aware UTC datetime, truncated to whole seconds (R1)."""
    return EPOCH + timedelta(seconds=ms // 1000)


def dt_to_ms(dt: Optional[datetime]) -> Optional[int]:
    if dt is None:
        return None
    return (as_utc(dt) - EPOCH) // timedelta(milliseconds=1)


def now_utc_seconds() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def parse_hhmm(value: str) -> time:
    h, m = map(int, value.split(":"))
    return time(h, m)


def is_overnight(emp: dict) -> bool:
    return emp["shift_end"] <= emp["shift_start"]


def attendance_date(punch_in: datetime, emp: dict) -> str:
    """R1: the IST calendar date of punch_in; for an overnight shift a punch-in
    before shift_end belongs to the shift that started the previous day."""
    local = punch_in.astimezone(IST)
    day = local.date()
    if is_overnight(emp) and local.time() < parse_hhmm(emp["shift_end"]):
        day -= timedelta(days=1)
    return day.isoformat()


def shift_bounds(date_str: str, emp: dict) -> tuple[datetime, datetime]:
    """Shift start/end instants (UTC) for the shift that starts on date_str (IST)."""
    day = date.fromisoformat(date_str)
    start = datetime.combine(day, parse_hhmm(emp["shift_start"]), IST)
    end = datetime.combine(day, parse_hhmm(emp["shift_end"]), IST)
    if is_overnight(emp):
        end += timedelta(days=1)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


def compute_late_minutes(punch_in: datetime, date_str: str, emp: dict) -> int:
    """R2: late only when strictly more than 10:00 after shift start; then whole
    minutes since shift start (not since the end of the grace period)."""
    start, _ = shift_bounds(date_str, emp)
    seconds = int((punch_in - start).total_seconds())
    return seconds // 60 if seconds > GRACE_SECONDS else 0


def round_half_up(value, places: int) -> float:
    """R8: decimal half-up rounding (Python's round() is banker's and float-based)."""
    quantum = Decimal(1).scaleb(-places)
    return float(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def compute_work_hours(punch_in: datetime, punch_out: datetime) -> float:
    """R4: seconds / 3600, 2 decimals half-up."""
    seconds = int((punch_out - punch_in).total_seconds())
    return round_half_up(Decimal(seconds) / Decimal(3600), 2)


def compute_overtime(punch_out: datetime, date_str: str, emp: dict) -> int:
    """R3: whole minutes after shift end, only when that is at least 30."""
    _, end = shift_bounds(date_str, emp)
    seconds = int((punch_out - end).total_seconds())
    minutes = seconds // 60 if seconds > 0 else 0
    return minutes if minutes >= MIN_OVERTIME_MINUTES else 0


def derived_fields(status: str, punch_in, punch_out, date_str: str, emp: dict) -> dict:
    """Everything the API computes for a record (DATA_MODEL.md, 'For a record your own code writes')."""
    if status not in PRESENCE_STATUSES or punch_in is None:
        return {"work_hours": None, "late_minutes": 0, "overtime_minutes": 0, "half_day": False}
    fields = {
        "work_hours": None,
        "late_minutes": compute_late_minutes(punch_in, date_str, emp),
        "overtime_minutes": 0,
        "half_day": False,
    }
    if punch_out is not None:
        hours = compute_work_hours(punch_in, punch_out)
        fields["work_hours"] = hours
        fields["overtime_minutes"] = compute_overtime(punch_out, date_str, emp)
        fields["half_day"] = Decimal(str(hours)) < HALF_DAY_BELOW_HOURS  # R5 uses the rounded value
    return fields


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def check_calendar_date(value: str) -> str:
    date.fromisoformat(value)  # rejects 2026-02-30 etc. (ValueError -> 422)
    return value


DATE_PATTERN = r"^\d{4}-\d{2}-\d{2}$"
MONTH_PATTERN = r"^\d{4}-(0[1-9]|1[0-2])$"
HHMM_PATTERN = r"^([01]\d|2[0-3]):[0-5]\d$"

DateStr = Annotated[str, Field(pattern=DATE_PATTERN), AfterValidator(check_calendar_date)]
HHMM = Annotated[str, Field(pattern=HHMM_PATTERN)]
Page = Annotated[int, Query(ge=1)]
PageSize = Annotated[int, Query(ge=1, le=100)]


def unprocessable(msg: str, loc: tuple = ("body",)) -> RequestValidationError:
    """A 422 with FastAPI's default validation body ({"detail": [...]}) for rule checks
    that can only run after reading the database."""
    return RequestValidationError([{"type": "value_error", "loc": loc, "msg": msg, "input": None}])


def validate_date_param(value: Optional[str], name: str) -> Optional[str]:
    if value is None:
        return None
    try:
        if not re.fullmatch(DATE_PATTERN, value):
            raise ValueError
        return check_calendar_date(value)
    except ValueError:
        raise unprocessable("must be a valid YYYY-MM-DD date", ("query", name))


def employee_out(doc: dict) -> dict:
    return {
        "emp_code": doc["emp_code"],
        "name": doc.get("name"),
        "email": doc.get("email"),
        "department": doc.get("department"),
        "shift_start": doc.get("shift_start"),
        "shift_end": doc.get("shift_end"),
        "joined_on": doc.get("joined_on"),
        "created_at": dt_to_ms(doc.get("created_at")),
    }


class EmployeeIn(BaseModel):
    emp_code: str = Field(pattern=r"^EMP\d{4,6}$")
    name: str = Field(min_length=1, max_length=100)
    email: str = Field(max_length=120, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    department: str = Field(min_length=1, max_length=50)
    shift_start: HHMM = "09:30"
    shift_end: HHMM = "18:30"
    joined_on: DateStr

    @model_validator(mode="after")
    def shift_times_differ(self):
        if self.shift_start == self.shift_end:
            raise ValueError("shift_start must differ from shift_end")
        return self


class PunchInIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None
    status: PresenceStatus = "PRESENT"


def history_out(entry: dict) -> dict:
    changes = {}
    for field, change in (entry.get("changes") or {}).items():
        if field in ("punch_in", "punch_out"):
            change = {"from": dt_to_ms(change.get("from")), "to": dt_to_ms(change.get("to"))}
        changes[field] = change
    return {"at": dt_to_ms(entry.get("at")), "by": entry.get("by"), "reason": entry.get("reason"), "changes": changes}


def record_out(doc: dict) -> dict:
    """Stored record -> API shape: no _id, epoch ms instants, defaults for legacy gaps."""
    return {
        "emp_code": doc["emp_code"],
        "date": doc["date"],
        "status": doc["status"],
        "punch_in": dt_to_ms(doc.get("punch_in")),
        "punch_out": dt_to_ms(doc.get("punch_out")),
        "work_hours": doc.get("work_hours"),
        "late_minutes": doc.get("late_minutes") or 0,
        "overtime_minutes": doc.get("overtime_minutes") or 0,
        "half_day": bool(doc.get("half_day", False)),
        "history": [history_out(h) for h in doc.get("history") or []],
    }


def get_employee_or_404(emp_code: str) -> dict:
    emp = db.employees.find_one({"emp_code": emp_code}, {"_id": 0})
    if emp is None:
        raise HTTPException(404, "employee not found")
    return emp


# --------------------------------------------------------------------------- #
# Endpoints provided
# --------------------------------------------------------------------------- #
@app.get("/health")
def health():
    try:
        client.admin.command("ping")
    except PyMongoError:
        return JSONResponse(status_code=503, content={"detail": "MongoDB is not reachable"})
    return {"status": "ok"}


@app.post("/employees", status_code=201)
def create_employee(body: EmployeeIn):
    doc = body.model_dump()
    doc["created_at"] = datetime.now(timezone.utc)
    try:
        # No find-then-insert: the unique index decides, so concurrent creates get one 201.
        db.employees.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "emp_code already exists")
    return employee_out(doc)


@app.get("/employees")
def list_employees(department: Optional[str] = None, page: Page = 1, page_size: PageSize = 20):
    q = {}
    if department is not None:
        q["department"] = department
    total = db.employees.count_documents(q)
    cursor = (
        db.employees.find(q, {"_id": 0})
        .sort("emp_code", ASCENDING)
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    return {"items": [employee_out(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


@app.post("/attendance/punch-in", status_code=201)
def punch_in(body: PunchInIn):
    emp = get_employee_or_404(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc_seconds()
    d = attendance_date(ts, emp)
    doc = {
        "emp_code": body.emp_code,
        "date": d,
        "status": body.status,
        "punch_in": ts,
        "punch_out": None,
        **derived_fields(body.status, ts, None, d, emp),
        "history": [],
    }
    try:
        # The unique (emp_code, date) index is the only arbiter: of N simultaneous
        # inserts exactly one succeeds, the rest raise DuplicateKeyError.
        db.attendance_logs.insert_one(doc)
    except DuplicateKeyError:
        raise HTTPException(409, "already punched in for this date")
    return record_out(doc)


ATTENDANCE_SORT = [("date", DESCENDING), ("emp_code", ASCENDING)]


def attendance_list_query(emp_code, date_from, date_to, status, page, page_size) -> dict:
    """Filter, sort and pagination for GET /attendance (shared with /admin/explain)."""
    date_from = validate_date_param(date_from, "date_from")
    date_to = validate_date_param(date_to, "date_to")
    if date_from and date_to and date_from > date_to:
        raise unprocessable("date_from must be on or before date_to", ("query", "date_from"))
    q = {}
    if emp_code is not None:
        q["emp_code"] = emp_code
    if date_from or date_to:
        q["date"] = {}
        if date_from:
            q["date"]["$gte"] = date_from
        if date_to:
            q["date"]["$lte"] = date_to
    if status is not None:
        q["status"] = status
    return {"filter": q, "sort": ATTENDANCE_SORT, "skip": (page - 1) * page_size, "limit": page_size}


@app.get("/attendance")
def list_attendance(
    emp_code: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    status: Optional[Status] = None,
    page: Page = 1,
    page_size: PageSize = 20,
):
    spec = attendance_list_query(emp_code, date_from, date_to, status, page, page_size)
    total = db.attendance_logs.count_documents(spec["filter"])
    cursor = (
        db.attendance_logs.find(spec["filter"], {"_id": 0})
        .sort(spec["sort"])
        .skip(spec["skip"])
        .limit(spec["limit"])
    )
    return {"items": [record_out(d) for d in cursor], "total": total, "page": page, "page_size": page_size}


# --------------------------------------------------------------------------- #
# TODO - the rest of the contract (see openapi.yaml):
#   POST  /attendance/punch-out
#   PATCH /attendance/{emp_code}/{date}
#   GET   /analytics/employees/{emp_code}/monthly
#   GET   /analytics/departments/summary
#   GET   /analytics/leaderboard/late
#   GET   /analytics/departments/{department}/trend
#   GET   /admin/explain/{endpoint}
# --------------------------------------------------------------------------- #
