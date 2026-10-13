# Employee Attendance & Analytics API

FastAPI + MongoDB implementation of the HROne assignment contract (`openapi.yaml`, data model in `DATA_MODEL.md`).
All application code is in `app/main.py`.

## Run
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                     # set MONGO_URI / MONGO_DB (or export them)
python sample_seed.py                                    # optional: the nine sample documents
uvicorn app.main:app --port 8000
```
* `MONGO_URI` and `MONGO_DB` come from the environment. A local `.env` is only a fallback, and real environment variables win.
* All indexes are created at startup (idempotently). `/health` returns 503 until MongoDB answers.
* Requires MongoDB 6.0+ (uses `$densify`, `$setWindowFields`, and `$lookup` with `localField` + `pipeline`). I tested on Atlas (MongoDB 8.0).

## Status
Everything in the contract is implemented: employees, punch-in/out, list, regularization, the four analytics
endpoints and `/admin/explain/{endpoint}`. Nothing is knowingly missing.

## How I tested it
```bash
pip install -r requirements-dev.txt
python tests/smoke_api.py          # rules R1-R5, status codes, concurrent creates/punches/patches
python tests/verify_analytics.py   # ~80k-record awkward dataset vs. a plain-Python reference + explain plans
```
Both scripts write to their own databases (`attendance_smoke`, `attendance_verify`), never to `MONGO_DB`.

## Notes / interpretations
* The trend range limit is 92 days inclusive, so `from=2026-07-01&to=2026-09-30` is accepted and one more day is a 422.
* `moving_avg_7d` averages the reported (4-decimal) `attendance_rate` values of the window.
* Rule violations that can only be checked after a database read (e.g. punch-out before punch-in) return 422
  with FastAPI's standard validation body (`detail` is a list).
* Half-up rounding is done with `Decimal` in Python and with Decimal128 inside pipelines, because `$round` uses banker's rounding.
* Defects found in the starter are in `REVIEW.md`; design answers are in `DECISIONS.md`.
