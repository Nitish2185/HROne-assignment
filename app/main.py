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
from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import AfterValidator, BaseModel, Field, model_validator
from pymongo import ASCENDING, DESCENDING, IndexModel, MongoClient, ReturnDocument
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


class PunchOutIn(BaseModel):
    emp_code: str
    punched_at: Optional[EpochMs] = None


@app.post("/attendance/punch-out")
def punch_out(body: PunchOutIn):
    emp = get_employee_or_404(body.emp_code)
    ts = ms_to_dt(body.punched_at) if body.punched_at is not None else now_utc_seconds()
    # Most recent record that started at or before this punch-out; works across midnight
    # because it is found by instant, not by calendar date.
    rec = db.attendance_logs.find_one(
        {"emp_code": body.emp_code, "punch_in": {"$lte": ts}},
        sort=[("punch_in", DESCENDING)],
    )
    if rec is None:
        raise HTTPException(404, "no punch-in found")
    if rec.get("punch_out") is not None:
        raise HTTPException(409, "already punched out")
    started = as_utc(rec["punch_in"])
    if ts <= started:
        raise unprocessable("punched_at must be after punch_in", ("body", "punched_at"))
    if ts - started > MAX_SHIFT:
        raise unprocessable("punched_at must be within 24 hours of punch_in", ("body", "punched_at"))

    fields = {"punch_out": ts, **derived_fields(rec["status"], started, ts, rec["date"], emp)}
    # Conditional update: only the request that still sees punch_out == null wins.
    updated = db.attendance_logs.find_one_and_update(
        {"_id": rec["_id"], "punch_out": None},
        {"$set": fields},
        projection={"_id": 0},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "already punched out")
    return record_out(updated)


class RegularizeIn(BaseModel):
    # Derived fields (work_hours, late_minutes, ...) are not declared, so they are ignored.
    status: Optional[Status] = None
    punch_in: Optional[EpochMs] = None
    punch_out: Optional[EpochMs] = None
    reason: str = Field(min_length=5, max_length=200)
    regularized_by: str = Field(min_length=1, max_length=50)

    @model_validator(mode="after")
    def no_explicit_nulls(self):
        for name in ("status", "punch_in", "punch_out"):
            if name in self.model_fields_set and getattr(self, name) is None:
                raise ValueError(f"{name} can not be null")
        return self


TRACKED_FIELDS = ("status", "punch_in", "punch_out", "work_hours", "late_minutes", "overtime_minutes", "half_day")


def current_values(rec: dict) -> dict:
    """The tracked fields of a stored record, with the documented defaults for gaps."""
    return {
        "status": rec["status"],
        "punch_in": as_utc(rec.get("punch_in")),
        "punch_out": as_utc(rec.get("punch_out")),
        "work_hours": rec.get("work_hours"),
        "late_minutes": rec.get("late_minutes") or 0,
        "overtime_minutes": rec.get("overtime_minutes") or 0,
        "half_day": bool(rec.get("half_day", False)),
    }


@app.patch("/attendance/{emp_code}/{date}")
def regularize_attendance(emp_code: str, body: RegularizeIn, record_date: Annotated[str, Path(alias="date")]):
    validate_date_param(record_date, "date")
    emp = get_employee_or_404(emp_code)
    rec = db.attendance_logs.find_one({"emp_code": emp_code, "date": record_date})
    if rec is None:
        raise HTTPException(404, "no attendance record for that date")
    old = current_values(rec)

    status = body.status or old["status"]
    if status in ABSENCE_STATUSES:
        if body.punch_in is not None or body.punch_out is not None:
            raise unprocessable(f"{status} clears both punch times; do not send punch_in/punch_out")
        new_in, new_out = None, None
    else:
        new_in = ms_to_dt(body.punch_in) if body.punch_in is not None else old["punch_in"]
        new_out = ms_to_dt(body.punch_out) if body.punch_out is not None else old["punch_out"]
        if new_in is None:
            raise unprocessable(f"status {status} requires a punch_in", ("body", "punch_in"))
        if attendance_date(new_in, emp) != record_date:
            raise unprocessable("punch_in must stay on the record's attendance date", ("body", "punch_in"))
        if new_out is not None and not (new_in < new_out <= new_in + MAX_SHIFT):
            raise unprocessable("punch_out must be after punch_in and within 24 hours", ("body", "punch_out"))

    new = {"status": status, "punch_in": new_in, "punch_out": new_out,
           **derived_fields(status, new_in, new_out, record_date, emp)}
    changes = {f: {"from": old[f], "to": new[f]} for f in TRACKED_FIELDS if old[f] != new[f]}
    if not changes:
        raise unprocessable("the request does not change anything")

    entry = {"at": datetime.now(timezone.utc), "by": body.regularized_by, "reason": body.reason, "changes": changes}
    # Optimistic concurrency: the update only applies if the record still has the values
    # (and the number of history entries) we read. A concurrent editor that got there
    # first makes this match nothing -> 409, so no history entry is silently lost.
    guard = {
        "_id": rec["_id"],
        "status": rec["status"],
        "punch_in": rec.get("punch_in"),
        "punch_out": rec.get("punch_out"),
        "$expr": {"$eq": [{"$size": {"$ifNull": ["$history", []]}}, len(rec.get("history") or [])]},
    }
    updated = db.attendance_logs.find_one_and_update(
        guard,
        {"$set": new, "$push": {"history": entry}},
        projection={"_id": 0},
        return_document=ReturnDocument.AFTER,
    )
    if updated is None:
        raise HTTPException(409, "the record was changed concurrently; retry")
    return record_out(updated)


# --------------------------------------------------------------------------- #
# Analytics - shared aggregation expressions. They read the stored derived values
# (never recompute them from punch times) and treat missing fields as their defaults.
# --------------------------------------------------------------------------- #
Month = Annotated[str, Query(pattern=MONTH_PATTERN)]


def weekday_expr(date_field: str) -> dict:
    """True when a YYYY-MM-DD string falls on Mon-Fri (R7)."""
    day = {"$dateFromString": {"dateString": date_field, "format": "%Y-%m-%d"}}
    return {"$lte": [{"$isoDayOfWeek": day}, 5]}


IS_PRESENT = {"$in": ["$status", list(PRESENCE_STATUSES)]}
# R5/R6: a presence record is worth 1, or 0.5 when it is a half day.
PRESENCE_VALUE = {"$cond": [IS_PRESENT, {"$cond": [{"$eq": [{"$ifNull": ["$half_day", False]}, True]}, 0.5, 1]}, 0]}
# present_days only counts Mon-Fri records (R7).
PRESENT_DAY_VALUE = {"$cond": [weekday_expr("$date"), PRESENCE_VALUE, 0]}
LATE_MINUTES = {"$ifNull": ["$late_minutes", 0]}
IS_LATE = {"$gt": [LATE_MINUTES, 0]}


def count_if(condition) -> dict:
    return {"$sum": {"$cond": [condition, 1, 0]}}


def round_half_up_expr(expr, places: int) -> dict:
    """R8 inside the pipeline: Decimal128 arithmetic, then floor(x * 10^n + 0.5) / 10^n.
    ($round is banker's rounding.) Null in, null out."""
    scale = 10 ** places
    scaled = {"$multiply": [{"$toDecimal": expr}, scale]}
    return {"$toDouble": {"$divide": [{"$floor": {"$add": [scaled, 0.5]}}, scale]}}


def month_bounds(month: str) -> tuple[str, str]:
    first = date.fromisoformat(f"{month}-01")
    nxt = (first.replace(day=28) + timedelta(days=4)).replace(day=1)
    return first.isoformat(), (nxt - timedelta(days=1)).isoformat()


def count_working_days(start: date, end: date) -> int:
    """Mon-Fri days in [start, end] (R7); 0 when start > end."""
    if start > end:
        return 0
    days = (end - start).days + 1
    full_weeks, rest = divmod(days, 7)
    return full_weeks * 5 + sum(1 for i in range(rest) if (start + timedelta(days=i)).weekday() < 5)


def employee_monthly_pipeline(emp_code: str, month: str) -> list:
    first, last = month_bounds(month)
    return [
        {"$match": {"emp_code": emp_code, "date": {"$gte": first, "$lte": last}}},
        {"$group": {
            "_id": None,
            "present_days": {"$sum": PRESENT_DAY_VALUE},
            "leave_days": count_if({"$eq": ["$status", "LEAVE"]}),
            "late_count": count_if(IS_LATE),
            "total_late_minutes": {"$sum": LATE_MINUTES},
            "total_overtime_minutes": {"$sum": {"$ifNull": ["$overtime_minutes", 0]}},
        }},
    ]


@app.get("/analytics/employees/{emp_code}/monthly")
def employee_monthly(emp_code: str, month: Month):
    emp = get_employee_or_404(emp_code)
    rows = list(db.attendance_logs.aggregate(employee_monthly_pipeline(emp_code, month)))
    stats = rows[0] if rows else {}

    first, last = month_bounds(month)
    joined = date.fromisoformat(emp["joined_on"])
    working_days = count_working_days(max(date.fromisoformat(first), joined), date.fromisoformat(last))
    present_days = stats.get("present_days", 0)
    pct = None
    if working_days:
        pct = round_half_up(Decimal(str(present_days)) * 100 / Decimal(working_days), 2)
    return {
        "emp_code": emp_code,
        "month": month,
        "working_days": working_days,
        "present_days": round_half_up(present_days, 2),
        "leave_days": stats.get("leave_days", 0),
        "late_count": stats.get("late_count", 0),
        "total_late_minutes": stats.get("total_late_minutes", 0),
        "total_overtime_minutes": stats.get("total_overtime_minutes", 0),
        "attendance_pct": pct,
    }


def department_summary_pipeline(month: str, department: Optional[str]) -> list:
    """Runs on `employees` so that headcount includes people with zero logs (R9).
    Each employee pulls in only their own logs for the month through an indexed
    $lookup on (emp_code, date), pre-aggregated to one row per employee."""
    first, last = month_bounds(month)
    match = {"joined_on": {"$lte": last}}
    if department is not None:
        match["department"] = department
    has_hours = {"$and": [IS_PRESENT, {"$isNumber": "$work_hours"}]}
    return [
        {"$match": match},
        {"$lookup": {
            "from": "attendance_logs",
            "localField": "emp_code",
            "foreignField": "emp_code",
            "pipeline": [
                {"$match": {"date": {"$gte": first, "$lte": last}}},
                {"$group": {
                    "_id": None,
                    "present_days": {"$sum": PRESENT_DAY_VALUE},
                    # Sum + count (not an average) so the department average is over records.
                    "work_hours_sum": {"$sum": {"$cond": [has_hours, {"$toDecimal": "$work_hours"}, 0]}},
                    "work_hours_count": count_if(has_hours),
                    "late_count": count_if(IS_LATE),
                    "total_late_minutes": {"$sum": LATE_MINUTES},
                    "leave_count": count_if({"$eq": ["$status", "LEAVE"]}),
                    "on_duty_count": count_if({"$eq": ["$status", "ON_DUTY"]}),
                }},
            ],
            "as": "stats",
        }},
        # Employees with no logs keep their row (stats missing -> $sum treats as 0).
        {"$unwind": {"path": "$stats", "preserveNullAndEmptyArrays": True}},
        {"$group": {
            "_id": "$department",
            "headcount": {"$sum": 1},
            "present_days": {"$sum": "$stats.present_days"},
            "work_hours_sum": {"$sum": "$stats.work_hours_sum"},
            "work_hours_count": {"$sum": "$stats.work_hours_count"},
            "late_count": {"$sum": "$stats.late_count"},
            "total_late_minutes": {"$sum": "$stats.total_late_minutes"},
            "leave_count": {"$sum": "$stats.leave_count"},
            "on_duty_count": {"$sum": "$stats.on_duty_count"},
        }},
        {"$sort": {"_id": 1}},
        {"$project": {
            "_id": 0,
            "department": "$_id",
            "headcount": "$headcount",
            "present_days": round_half_up_expr("$present_days", 2),
            "avg_work_hours": {"$cond": [
                {"$gt": ["$work_hours_count", 0]},
                round_half_up_expr({"$divide": ["$work_hours_sum", "$work_hours_count"]}, 2),
                None,
            ]},
            "late_count": "$late_count",
            "total_late_minutes": "$total_late_minutes",
            "leave_count": "$leave_count",
            "on_duty_count": "$on_duty_count",
        }},
    ]


@app.get("/analytics/departments/summary")
def department_summary(month: Month, department: Optional[str] = None):
    items = list(db.employees.aggregate(department_summary_pipeline(month, department)))
    return {"month": month, "items": items}


LeaderboardLimit = Annotated[int, Query(ge=1, le=50)]


def late_leaderboard_pipeline(month: str, limit: int, department: Optional[str]) -> list:
    first, last = month_bounds(month)
    pipeline = [
        {"$match": {"date": {"$gte": first, "$lte": last}, "late_minutes": {"$gt": 0}}},
        {"$group": {"_id": "$emp_code", "total_late_minutes": {"$sum": "$late_minutes"}, "late_count": {"$sum": 1}}},
        # Inner join: logs whose employee does not exist are dropped before ranking.
        {"$lookup": {
            "from": "employees",
            "localField": "_id",
            "foreignField": "emp_code",
            "pipeline": [{"$project": {"_id": 0, "name": 1, "department": 1}}],
            "as": "emp",
        }},
        {"$unwind": "$emp"},
    ]
    if department is not None:
        pipeline.append({"$match": {"emp.department": department}})  # rank inside the department
    pipeline += [
        # $rank is standard competition ranking: ties share a rank, the next one is skipped.
        {"$setWindowFields": {"sortBy": {"total_late_minutes": -1}, "output": {"rank": {"$rank": {}}}}},
        # limit applies to the rank, not to the number of rows, so ties at the cutoff all stay.
        {"$match": {"rank": {"$lte": limit}}},
        {"$sort": {"total_late_minutes": -1, "_id": 1}},
        {"$project": {
            "_id": 0,
            "rank": "$rank",
            "emp_code": "$_id",
            "name": "$emp.name",
            "department": "$emp.department",
            "total_late_minutes": "$total_late_minutes",
            "late_count": "$late_count",
        }},
    ]
    return pipeline


@app.get("/analytics/leaderboard/late")
def late_leaderboard(month: Month, limit: LeaderboardLimit = 10, department: Optional[str] = None):
    items = list(db.attendance_logs.aggregate(late_leaderboard_pipeline(month, limit, department)))
    return {"month": month, "items": items}


MAX_TREND_DAYS = 92


def validate_trend_range(date_from: Optional[str], date_to: Optional[str]) -> tuple[str, str]:
    if date_from is None or date_to is None:
        raise unprocessable("from and to are required", ("query", "from" if date_from is None else "to"))
    start = validate_date_param(date_from, "from")
    end = validate_date_param(date_to, "to")
    if end < start:
        raise unprocessable("to must be on or after from", ("query", "to"))
    if (date.fromisoformat(end) - date.fromisoformat(start)).days + 1 > MAX_TREND_DAYS:
        raise unprocessable(f"the range can not exceed {MAX_TREND_DAYS} days", ("query", "to"))
    return start, end


def department_trend_pipeline(department: str, start: str, end: str) -> list:
    """One row per calendar day in [start, end], computed entirely in MongoDB.

    Each employee of the department emits "events": +1 headcount on the day they join
    (or on `start` if they joined earlier) and one event per log in the range. Events are
    summed per day, missing days are created with $densify, and a running sum turns the
    join events into the headcount of each day (R9)."""
    lower = datetime.combine(date.fromisoformat(start), time(), timezone.utc)
    upper = datetime.combine(date.fromisoformat(end) + timedelta(days=1), time(), timezone.utc)  # exclusive
    rate = {"$cond": [
        {"$and": ["$is_working_day", {"$gt": ["$headcount", 0]}]},
        round_half_up_expr({"$divide": ["$present_count", "$headcount"]}, 4),
        None,
    ]}
    return [
        {"$match": {"department": department}},
        {"$lookup": {
            "from": "attendance_logs",
            "localField": "emp_code",
            "foreignField": "emp_code",
            "pipeline": [
                {"$match": {"date": {"$gte": start, "$lte": end}}},
                {"$project": {
                    "_id": 0,
                    "date": 1,
                    "joins": {"$literal": 0},
                    "present": PRESENCE_VALUE,  # any day of the week
                    "late": {"$cond": [IS_LATE, 1, 0]},
                }},
            ],
            "as": "log_events",
        }},
        {"$project": {"_id": 0, "events": {"$concatArrays": [
            "$log_events",
            {"$cond": [
                {"$lte": ["$joined_on", end]},
                [{"date": {"$max": ["$joined_on", start]}, "joins": 1, "present": 0, "late": 0}],
                # Not joined yet: still emit an empty event so $densify always has input.
                [{"date": start, "joins": 0, "present": 0, "late": 0}],
            ]},
        ]}}},
        {"$unwind": "$events"},
        {"$group": {
            "_id": "$events.date",
            "joins": {"$sum": "$events.joins"},
            "present_count": {"$sum": "$events.present"},
            "late_count": {"$sum": "$events.late"},
        }},
        {"$set": {"day": {"$dateFromString": {"dateString": "$_id", "format": "%Y-%m-%d"}}}},
        # Gap filling: one document per calendar day, weekends and empty days included.
        {"$densify": {"field": "day", "range": {
            "step": 1, "unit": "day", "bounds": [lower, upper],
        }}},
        {"$setWindowFields": {
            "sortBy": {"day": 1},
            "output": {"headcount": {"$sum": {"$ifNull": ["$joins", 0]}, "window": {"documents": ["unbounded", "current"]}}},
        }},
        {"$set": {
            "present_count": {"$ifNull": ["$present_count", 0]},
            "late_count": {"$ifNull": ["$late_count", 0]},
            "is_working_day": {"$lte": [{"$isoDayOfWeek": "$day"}, 5]},
        }},
        {"$set": {"attendance_rate": rate}},
        # Trailing 7-row window inside the requested range; $avg skips nulls and
        # returns null when the window holds none.
        {"$setWindowFields": {
            "sortBy": {"day": 1},
            "output": {"moving_avg_7d": {"$avg": "$attendance_rate", "window": {"documents": [-6, "current"]}}},
        }},
        {"$project": {
            "_id": 0,
            "date": {"$dateToString": {"date": "$day", "format": "%Y-%m-%d"}},
            "is_working_day": "$is_working_day",
            "headcount": "$headcount",
            "present_count": {"$toDouble": "$present_count"},
            "late_count": "$late_count",
            "attendance_rate": "$attendance_rate",
            "moving_avg_7d": round_half_up_expr("$moving_avg_7d", 4),
        }},
    ]


@app.get("/analytics/departments/{department}/trend")
def department_trend(
    department: str,
    date_from: Annotated[Optional[str], Query(alias="from")] = None,
    date_to: Annotated[Optional[str], Query(alias="to")] = None,
):
    start, end = validate_trend_range(date_from, date_to)
    if db.employees.find_one({"department": department}, {"_id": 1}) is None:
        raise HTTPException(404, "unknown department")
    items = list(db.employees.aggregate(department_trend_pipeline(department, start, end)))
    return {"department": department, "items": items}


# --------------------------------------------------------------------------- #
# TODO - the rest of the contract (see openapi.yaml):
#   GET   /admin/explain/{endpoint}
# --------------------------------------------------------------------------- #
