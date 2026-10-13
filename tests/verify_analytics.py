"""Seeds a large, deliberately awkward dataset into a separate database and compares every
analytics endpoint with a plain-Python reference implementation of the rules.

Usage:  python tests/verify_analytics.py            (uses MONGO_URI; MONGO_DB is forced to
        attendance_verify so your dev data is untouched)
        python tests/verify_analytics.py --reuse    (skip re-seeding)

Cases covered: mid-month joiners, joiners after the period, employees with zero logs, weekend
records, half days, open records, legacy documents without half_day/history/late_minutes,
orphan logs (no employee), tied late totals, overnight shifts, a case-variant department.
"""
import json
import os
import random
import sys
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from decimal import ROUND_HALF_UP, Decimal

os.environ["MONGO_DB"] = "attendance_verify"  # real env var wins over .env
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient  # noqa: E402

from app.main import app, db  # noqa: E402

RNG = random.Random(42)
DEPARTMENTS = ["Engineering", "Sales", "Support", "HR", "engineering"]
FIRST_DAY, LAST_DAY = date(2026, 4, 1), date(2026, 8, 20)
PRESENCE = {"PRESENT", "WFH", "ON_DUTY"}


def q(value, places):
    return float(Decimal(str(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP))


# --------------------------------------------------------------------------- seed
def make_employees(n=1000):
    emps = []
    for i in range(n):
        joined = date(2025, 6, 1) + timedelta(days=RNG.randint(0, 450))  # some join after Aug 2026
        overnight = RNG.random() < 0.1
        emps.append({
            "emp_code": f"EMP{i + 1:04d}",
            "name": f"Person {i + 1}",
            "email": f"p{i + 1}@example.com",
            "department": RNG.choice(DEPARTMENTS),
            "shift_start": "22:00" if overnight else "09:30",
            "shift_end": "06:00" if overnight else "18:30",
            "joined_on": joined.isoformat(),
            "created_at": datetime(2025, 6, 1, tzinfo=timezone.utc),
        })
    return emps


def make_logs(emps):
    logs = []
    silent = {e["emp_code"] for e in RNG.sample(emps, 40)}  # zero logs at all
    codes = [e["emp_code"] for e in emps if e["emp_code"] not in silent]
    codes += [f"EMP9{i:04d}" for i in range(15)]  # orphans
    day = FIRST_DAY
    while day <= LAST_DAY:
        weekend = day.weekday() >= 5
        for code in codes:
            if RNG.random() > (0.08 if weekend else 0.78):
                continue
            status = RNG.choices(["PRESENT", "WFH", "ON_DUTY", "ABSENT", "LEAVE"], [60, 15, 8, 9, 8])[0]
            doc = {"emp_code": code, "date": day.isoformat(), "status": status}
            if status in PRESENCE:
                start = datetime(day.year, day.month, day.day, 4, 0, tzinfo=timezone.utc)
                doc["punch_in"] = start + timedelta(seconds=RNG.randint(0, 7200))
                late = RNG.choice([0, 0, 0, 0, 11, 15, 20, 35, 45])
                if RNG.random() < 0.9:
                    doc["late_minutes"] = late
                if RNG.random() < 0.08:  # still open
                    doc.update(punch_out=None, work_hours=None, overtime_minutes=0, half_day=False)
                else:
                    hours = q(RNG.uniform(2.0, 10.5), 2)
                    doc["punch_out"] = doc["punch_in"] + timedelta(hours=hours)
                    doc["work_hours"] = hours
                    doc["overtime_minutes"] = RNG.choice([0, 0, 0, 30, 45, 61])
                    if RNG.random() < 0.9:  # legacy docs omit half_day
                        doc["half_day"] = hours < 4.5
            else:
                doc.update(punch_in=None, punch_out=None, work_hours=None, late_minutes=0, overtime_minutes=0,
                           half_day=False)
            if RNG.random() < 0.9:
                doc["history"] = []
            logs.append(doc)
        day += timedelta(days=1)
    return logs


def seed():
    emps = make_employees()
    logs = make_logs(emps)
    db.employees.drop()
    db.attendance_logs.drop()
    db.employees.insert_many(emps)
    for i in range(0, len(logs), 10000):
        db.attendance_logs.insert_many(logs[i:i + 10000], ordered=False)
    print(f"seeded {len(emps)} employees, {len(logs)} logs into {db.name}")


# --------------------------------------------------------------------------- reference
def load():
    emps = {e["emp_code"]: e for e in db.employees.find({}, {"_id": 0})}
    logs = list(db.attendance_logs.find({}, {"_id": 0, "history": 0, "punch_in": 0, "punch_out": 0}))
    return emps, logs


def month_range(month):
    first = date.fromisoformat(month + "-01")
    last = (first.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
    return first, last


def is_weekday(s):
    return date.fromisoformat(s).weekday() < 5


def presence_value(log):
    if log["status"] not in PRESENCE:
        return 0
    return 0.5 if log.get("half_day", False) else 1


def ref_monthly(emp, logs, month):
    first, last = month_range(month)
    mine = [l for l in logs if l["emp_code"] == emp["emp_code"] and first.isoformat() <= l["date"] <= last.isoformat()]
    start = max(first, date.fromisoformat(emp["joined_on"]))
    wd = sum(1 for i in range((last - start).days + 1) if (start + timedelta(days=i)).weekday() < 5) if start <= last else 0
    present = sum(presence_value(l) for l in mine if is_weekday(l["date"]))
    return {
        "emp_code": emp["emp_code"], "month": month, "working_days": wd, "present_days": q(present, 2),
        "leave_days": sum(1 for l in mine if l["status"] == "LEAVE"),
        "late_count": sum(1 for l in mine if (l.get("late_minutes") or 0) > 0),
        "total_late_minutes": sum(l.get("late_minutes") or 0 for l in mine),
        "total_overtime_minutes": sum(l.get("overtime_minutes") or 0 for l in mine),
        "attendance_pct": q(Decimal(str(present)) * 100 / wd, 2) if wd else None,
    }


def ref_summary(emps, logs, month, department=None):
    first, last = month_range(month)
    members = defaultdict(set)
    for e in emps.values():
        if e["joined_on"] <= last.isoformat() and (department is None or e["department"] == department):
            members[e["department"]].add(e["emp_code"])
    items = []
    for dept in sorted(members):
        mine = [l for l in logs if l["emp_code"] in members[dept] and first.isoformat() <= l["date"] <= last.isoformat()]
        wh = [Decimal(str(l["work_hours"])) for l in mine if l["status"] in PRESENCE and l.get("work_hours") is not None]
        items.append({
            "department": dept, "headcount": len(members[dept]),
            "present_days": q(sum(presence_value(l) for l in mine if is_weekday(l["date"])), 2),
            "avg_work_hours": q(sum(wh) / len(wh), 2) if wh else None,
            "late_count": sum(1 for l in mine if (l.get("late_minutes") or 0) > 0),
            "total_late_minutes": sum(l.get("late_minutes") or 0 for l in mine),
            "leave_count": sum(1 for l in mine if l["status"] == "LEAVE"),
            "on_duty_count": sum(1 for l in mine if l["status"] == "ON_DUTY"),
        })
    return {"month": month, "items": items}


def ref_leaderboard(emps, logs, month, limit=10, department=None):
    first, last = month_range(month)
    totals = defaultdict(lambda: [0, 0])
    for l in logs:
        if first.isoformat() <= l["date"] <= last.isoformat() and (l.get("late_minutes") or 0) > 0 and l["emp_code"] in emps:
            if department is None or emps[l["emp_code"]]["department"] == department:
                totals[l["emp_code"]][0] += l["late_minutes"]
                totals[l["emp_code"]][1] += 1
    rows = sorted(totals.items(), key=lambda kv: (-kv[1][0], kv[0]))
    out = []
    for code, (total, count) in rows:
        rank = 1 + sum(1 for _, (t, _) in rows if t > total)
        if rank <= limit:
            e = emps[code]
            out.append({"rank": rank, "emp_code": code, "name": e["name"], "department": e["department"],
                        "total_late_minutes": total, "late_count": count})
    return {"month": month, "items": out}


def ref_trend(emps, logs, department, start, end):
    members = {c: e for c, e in emps.items() if e["department"] == department}
    by_day = defaultdict(list)
    for l in logs:
        if l["emp_code"] in members and start <= l["date"] <= end:
            by_day[l["date"]].append(l)
    rows, rates = [], []
    d, last = date.fromisoformat(start), date.fromisoformat(end)
    while d <= last:
        s = d.isoformat()
        hc = sum(1 for e in members.values() if e["joined_on"] <= s)
        present = sum(presence_value(l) for l in by_day[s])
        working = d.weekday() < 5
        rate = q(Decimal(str(present)) / hc, 4) if working and hc > 0 else None
        rates.append(rate)
        window = [r for r in rates[-7:] if r is not None]
        mavg = q(sum(Decimal(str(r)) for r in window) / len(window), 4) if window else None
        rows.append({"date": s, "is_working_day": working, "headcount": hc, "present_count": float(present),
                     "late_count": sum(1 for l in by_day[s] if (l.get("late_minutes") or 0) > 0),
                     "attendance_rate": rate, "moving_avg_7d": mavg})
        d += timedelta(days=1)
    return {"department": department, "items": rows}


# --------------------------------------------------------------------------- compare
def main():
    if "--reuse" not in sys.argv:
        seed()
    failures = 0
    with TestClient(app) as c:  # lifespan creates the indexes
        emps, logs = load()

        def check(label, got, want):
            nonlocal failures
            if got != want:
                failures += 1
                print("MISMATCH", label)
                print("  got ", json.dumps(got)[:600])
                print("  want", json.dumps(want)[:600])

        for month in ["2026-05", "2026-07", "2026-08", "2025-01"]:
            for code in RNG.sample(sorted(emps), 25):
                got = c.get(f"/analytics/employees/{code}/monthly", params={"month": month}).json()
                check(f"monthly {code} {month}", got, ref_monthly(emps[code], logs, month))
            check(f"summary {month}", c.get("/analytics/departments/summary", params={"month": month}).json(),
                  ref_summary(emps, logs, month))
            check(f"summary {month} Sales",
                  c.get("/analytics/departments/summary", params={"month": month, "department": "Sales"}).json(),
                  ref_summary(emps, logs, month, "Sales"))
            for limit, dept in [(10, None), (1, None), (3, "engineering"), (50, "HR")]:
                params = {"month": month, "limit": limit, **({"department": dept} if dept else {})}
                check(f"leaderboard {params}", c.get("/analytics/leaderboard/late", params=params).json(),
                      ref_leaderboard(emps, logs, month, limit, dept))
        for dept, start, end in [("Sales", "2026-07-01", "2026-07-31"), ("HR", "2026-05-15", "2026-08-14"),
                                 ("engineering", "2026-08-10", "2026-08-31"), ("Support", "2025-01-01", "2025-01-05")]:
            check(f"trend {dept} {start}..{end}",
                  c.get(f"/analytics/departments/{dept}/trend", params={"from": start, "to": end}).json(),
                  ref_trend(emps, logs, dept, start, end))

        explain_cases = [
            ("attendance_list", {}), ("attendance_list", {"emp_code": "EMP0007"}),
            ("attendance_list", {"status": "LEAVE", "page": 3}),
            ("attendance_list", {"date_from": "2026-07-01", "date_to": "2026-07-31", "status": "WFH"}),
            ("employee_monthly", {"emp_code": "EMP0007", "month": "2026-07"}),
            ("department_summary", {"month": "2026-07"}), ("department_summary", {"month": "2026-07", "department": "HR"}),
            ("late_leaderboard", {"month": "2026-07"}), ("late_leaderboard", {"month": "2026-07", "department": "HR"}),
            ("department_trend", {"department": "Sales", "from": "2026-07-01", "to": "2026-09-30"}),
        ]
        for endpoint, params in explain_cases:
            r = c.get(f"/admin/explain/{endpoint}", params=params)
            text = json.dumps(r.json())
            ok = r.status_code == 200 and "IXSCAN" in text and "COLLSCAN" not in text
            failures += not ok
            print(f"explain {endpoint:18} {params}: {'ok' if ok else 'FAIL'}")
    print("ALL ANALYTICS MATCH" if failures == 0 else f"{failures} FAILURES")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
