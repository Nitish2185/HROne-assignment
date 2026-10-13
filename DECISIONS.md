# DECISIONS.md

1. **Indexes.** `employees`: unique `emp_code` (no duplicate creates), `department+emp_code` (filtered list, trend), `joined_on` (summary headcount). `attendance_logs`: unique `emp_code+date` (one record per day, per-employee lookups), `date desc+emp_code` (list sort, month ranges), `status+date+emp_code` (status filter), `emp_code+punch_in desc` (punch-out). I rejected an index on `late_minutes`: the leaderboard already narrows by date, and the extra index would only slow down writes.

2. **Punch-in race.** Both requests validate, compute the same date and call `insert_one`. There is no read before the write. MongoDB's unique `(emp_code, date)` index lets one insert through. The other raises `DuplicateKeyError`, which I turn into a 409.

3. **Ties.** `$rank` gives tied employees the same rank and skips the next one. I filter on `rank <= limit`, so everyone tied at the cutoff is returned, even when that is more than `limit` rows.

4. **Headcount.** The pipeline starts from `employees` (`joined_on <= month end`) and uses `$lookup` to bring in each person's logs. `$unwind` with `preserveNullAndEmptyArrays` keeps people who have no logs.

5. **100x data.** Keep pre-aggregated daily per-department counters, updated on every write, so analytics no longer scan raw logs. I would also shard `attendance_logs` by `emp_code`.
