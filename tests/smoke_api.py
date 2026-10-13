"""Rule, status-code and race checks for the CRUD endpoints, against a scratch database.

Usage:  python tests/smoke_api.py      (uses MONGO_URI; MONGO_DB is forced to attendance_smoke)
"""
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

os.environ["MONGO_DB"] = "attendance_smoke"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app, client  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
failures = 0


def ms(iso_ist: str) -> int:
    """'2026-07-06T09:40:01.900' (IST) -> epoch ms."""
    return int(datetime.fromisoformat(iso_ist).replace(tzinfo=IST).timestamp() * 1000)


def expect(label, got, want):
    global failures
    ok = got == want
    failures += not ok
    print(f"{'ok  ' if ok else 'FAIL'} {label}: {got!r}" + ("" if ok else f" (want {want!r})"))


def parallel(fn, n=8):
    with ThreadPoolExecutor(n) as ex:
        return sorted(ex.map(lambda _: fn(), range(n)))


def main():
    client.drop_database("attendance_smoke")
    with TestClient(app) as c:
        emp = {"emp_code": "EMP0001", "name": "Day", "email": "d@x.io", "department": "Eng", "joined_on": "2026-07-01"}
        expect("create race", parallel(lambda: c.post("/employees", json=emp).status_code), [201] + [409] * 7)
        c.post("/employees", json={**emp, "emp_code": "EMP0002", "name": "Night", "shift_start": "22:00",
                                   "shift_end": "06:00"})
        expect("same shift times", c.post("/employees", json={**emp, "emp_code": "EMP0003", "shift_start": "09:00",
                                                               "shift_end": "09:00"}).status_code, 422)
        expect("bad emp_code", c.post("/employees", json={**emp, "emp_code": "E1"}).status_code, 422)
        page = c.get("/employees", params={"department": "Eng", "page": 1, "page_size": 1}).json()
        expect("employees total is filtered", (page["total"], len(page["items"])), (2, 1))

        pin = lambda body: c.post("/attendance/punch-in", json=body)
        pout = lambda body: c.post("/attendance/punch-out", json=body)

        # R1 truncation + R2 grace boundary
        r = pin({"emp_code": "EMP0001", "punched_at": ms("2026-07-06T09:40:00.900")}).json()
        expect("R2 09:40:00.900 -> 0 and truncated", (r["late_minutes"], r["punch_in"] % 1000), (0, 0))
        expect("punch-in race", parallel(lambda: pin({"emp_code": "EMP0001", "punched_at": ms("2026-07-07T09:40:01")}).status_code),
               [201] + [409] * 7)
        expect("R2 09:40:01 -> 10", c.get("/attendance", params={"emp_code": "EMP0001", "date_from": "2026-07-07"})
               .json()["items"][0]["late_minutes"], 10)
        r = pin({"emp_code": "EMP0001", "punched_at": ms("2026-07-08T10:15:59")}).json()
        expect("R2 10:15:59 -> 45", r["late_minutes"], 45)
        for bad in [1783312500, 1783312500000.0, "1783312500000", True]:
            expect(f"punched_at {bad!r}", pin({"emp_code": "EMP0001", "punched_at": bad}).status_code, 422)
        expect("status ABSENT at punch-in", pin({"emp_code": "EMP0001", "status": "ABSENT"}).status_code, 422)
        expect("unknown employee", pin({"emp_code": "EMP0999"}).status_code, 404)

        # Overnight: 01:00 on the 7th belongs to the shift of the 6th; out 06:40 -> 40 OT
        r = pin({"emp_code": "EMP0002", "punched_at": ms("2026-07-07T01:00:00")}).json()
        expect("R1 overnight date / late", (r["date"], r["late_minutes"]), ("2026-07-06", 180))
        out = parallel(lambda: pout({"emp_code": "EMP0002", "punched_at": ms("2026-07-07T06:40:00")}).status_code, 6)
        expect("punch-out race", out, [200] + [409] * 5)
        r = c.get("/attendance", params={"emp_code": "EMP0002"}).json()["items"][0]
        expect("R3/R4 overnight", (r["work_hours"], r["overtime_minutes"], r["half_day"]), (5.67, 40, False))

        # R4 half-up and R5 on the rounded value: 4h29m42s = 4.495 h -> 4.50 -> not a half day
        r = pout({"emp_code": "EMP0001", "punched_at": ms("2026-07-06T14:09:42")}).json()
        expect("R4/R5 4.495 -> 4.5, not half day", (r["work_hours"], r["half_day"]), (4.5, False))
        expect("already punched out", pout({"emp_code": "EMP0001", "punched_at": ms("2026-07-06T15:00:00")}).status_code, 409)
        expect("out > 24h", pout({"emp_code": "EMP0001", "punched_at": ms("2026-07-09T10:16:00")}).status_code, 422)
        expect("R3 29 min -> 0", pout({"emp_code": "EMP0001", "punched_at": ms("2026-07-07T18:59:59")}).json()
               ["overtime_minutes"], 0)

        patch = lambda d, body: c.patch(f"/attendance/EMP0001/{d}", json={"reason": "correction", "regularized_by": "hr", **body})
        r = patch("2026-07-08", {"punch_in": ms("2026-07-08T09:25:00"), "work_hours": 99}).json()
        expect("patch recompute + history", (r["late_minutes"], list(r["history"][0]["changes"])), (0, ["punch_in", "late_minutes"]))
        expect("patch no-op", patch("2026-07-08", {"punch_in": ms("2026-07-08T09:25:00")}).status_code, 422)
        expect("patch leave + punch", patch("2026-07-08", {"status": "LEAVE", "punch_out": ms("2026-07-08T18:00:00")}).status_code, 422)
        expect("patch punch_in other day", patch("2026-07-08", {"punch_in": ms("2026-07-09T09:25:00")}).status_code, 422)
        expect("patch missing record", patch("2026-07-20", {"status": "LEAVE"}).status_code, 404)
        r = patch("2026-07-08", {"status": "LEAVE"}).json()
        expect("patch to LEAVE clears", (r["punch_in"], r["work_hours"], len(r["history"])), (None, None, 2))
        codes = parallel(lambda: patch("2026-07-08", {"status": "PRESENT", "punch_in": ms("2026-07-08T09:31:00")}).status_code, 6)
        hist = len(c.get("/attendance", params={"emp_code": "EMP0001", "date_from": "2026-07-08", "date_to": "2026-07-08"})
                   .json()["items"][0]["history"])
        expect("concurrent patches keep every success in history", hist, 2 + codes.count(200))

        expect("date_from > date_to", c.get("/attendance", params={"date_from": "2026-07-09", "date_to": "2026-07-01"}).status_code, 422)
        expect("page_size 101", c.get("/attendance", params={"page_size": 101}).status_code, 422)
    client.drop_database("attendance_smoke")
    print("ALL CHECKS PASSED" if failures == 0 else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
