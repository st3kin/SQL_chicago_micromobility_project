TRUNCATE mds.status_changes, mds.incidents, mds.vehicles, mds.service_areas RESTART IDENTITY CASCADE;

-- service_areas
INSERT INTO mds.service_areas(area_id, area_name)
SELECT DISTINCT a.area_id, a.area_name
FROM (
  SELECT start_area_id AS area_id, start_area_name AS area_name FROM mds.trips_canon
  UNION
  SELECT end_area_id   AS area_id, end_area_name   AS area_name FROM mds.trips_canon
) a
WHERE a.area_id IS NOT NULL;

-- vehicles
WITH mapped AS (
  SELECT
    COALESCE(NULLIF(device_id,''), 'V' || lpad(mds.hash_mod(trip_id, 5000)::text, 5, '0')) AS vehicle_id,
    vendor, started_at, ended_at
  FROM mds.trips_canon
)
INSERT INTO mds.vehicles (vehicle_id, vendor, first_seen, last_seen)
SELECT vehicle_id, MAX(vendor) FILTER (WHERE vendor IS NOT NULL),
       MIN(started_at), MAX(ended_at)
FROM mapped
GROUP BY vehicle_id;

-- status_changes: trip_start
INSERT INTO mds.status_changes (vehicle_id, event_time, event_type, area_id, trip_id, note)
SELECT
  COALESCE(NULLIF(tc.device_id,''), 'V' || lpad(mds.hash_mod(tc.trip_id, 5000)::text, 5, '0')),
  tc.started_at, 'trip_start', tc.start_area_id, tc.trip_id, NULL
FROM mds.trips_canon tc
WHERE tc.started_at IS NOT NULL AND tc.start_area_id IS NOT NULL;

-- status_changes: trip_end
INSERT INTO mds.status_changes (vehicle_id, event_time, event_type, area_id, trip_id, note)
SELECT
  COALESCE(NULLIF(tc.device_id,''), 'V' || lpad(mds.hash_mod(tc.trip_id, 5000)::text, 5, '0')),
  tc.ended_at, 'trip_end', COALESCE(tc.end_area_id, tc.start_area_id), tc.trip_id, NULL
FROM mds.trips_canon tc
WHERE tc.ended_at IS NOT NULL;

-- battery_low
WITH base AS (
  SELECT tc.*, EXTRACT(EPOCH FROM (tc.ended_at - tc.started_at)) AS dur_sec
  FROM mds.trips_canon tc
)
INSERT INTO mds.status_changes (vehicle_id, event_time, event_type, area_id, trip_id, note)
SELECT
  COALESCE(NULLIF(b.device_id,''), 'V' || lpad(mds.hash_mod(b.trip_id, 5000)::text, 5, '0')),
  b.ended_at + make_interval(mins => 1),
  'battery_low',
  COALESCE(b.end_area_id, b.start_area_id),
  b.trip_id,
  CASE WHEN b.dur_sec >= 1800 THEN 'Long trip' ELSE NULL END
FROM base b
WHERE (mds.rand01(b.trip_id || '|bat') < 0.18) OR b.dur_sec >= 1800;

-- rebalance — pickup
WITH candidates AS (
  SELECT tc.*,
         mds.hash_mod(tc.trip_id || '|reb_area',
                      GREATEST(1,(SELECT COUNT(*)::int FROM mds.service_areas))) AS area_pick
  FROM mds.trips_canon tc
),
areas AS (
  SELECT area_id, row_number() OVER (ORDER BY area_id) AS rn FROM mds.service_areas
),
reb AS (
  SELECT
    c.*,
    (SELECT a.area_id FROM areas a WHERE a.rn = c.area_pick) AS drop_area_id,
    (mds.hash_mod(c.trip_id || '|reb_min1', 25) + 5)  AS mins_to_pickup,
    (mds.hash_mod(c.trip_id || '|reb_min2', 40) + 20) AS mins_to_dropoff
  FROM candidates c
  WHERE mds.rand01(c.trip_id || '|reb') < 0.20
)
INSERT INTO mds.status_changes (vehicle_id, event_time, event_type, area_id, trip_id, note)
SELECT
  COALESCE(NULLIF(r.device_id,''), 'V' || lpad(mds.hash_mod(r.trip_id, 5000)::text, 5, '0')),
  r.ended_at + make_interval(mins => r.mins_to_pickup),
  'rebalance_pickup',
  COALESCE(r.end_area_id, r.start_area_id),
  r.trip_id,
  'Post-trip rebalance'
FROM reb r;

-- rebalance — dropoff
WITH candidates AS (
  SELECT tc.*,
         mds.hash_mod(tc.trip_id || '|reb_area',
                      GREATEST(1,(SELECT COUNT(*)::int FROM mds.service_areas))) AS area_pick
  FROM mds.trips_canon tc
),
areas AS (
  SELECT area_id, row_number() OVER (ORDER BY area_id) AS rn FROM mds.service_areas
),
reb AS (
  SELECT
    c.*,
    (SELECT a.area_id FROM areas a WHERE a.rn = c.area_pick) AS drop_area_id,
    (mds.hash_mod(c.trip_id || '|reb_min1', 25) + 5)  AS mins_to_pickup,
    (mds.hash_mod(c.trip_id || '|reb_min2', 40) + 20) AS mins_to_dropoff
  FROM candidates c
  WHERE mds.rand01(c.trip_id || '|reb') < 0.20
)
INSERT INTO mds.status_changes (vehicle_id, event_time, event_type, area_id, trip_id, note)
SELECT
  COALESCE(NULLIF(r.device_id,''), 'V' || lpad(mds.hash_mod(r.trip_id, 5000)::text, 5, '0')),
  r.ended_at + make_interval(mins => r.mins_to_pickup + r.mins_to_dropoff),
  'rebalance_dropoff',
  COALESCE(r.drop_area_id, COALESCE(r.end_area_id, r.start_area_id)),
  r.trip_id,
  'Post-trip rebalance'
FROM reb r;

-- maintenance — pickup
WITH m AS (
  SELECT tc.*,
         (mds.hash_mod(tc.trip_id || '|mnt_min1', 120) + 60) AS mins_to_pickup,
         (mds.hash_mod(tc.trip_id || '|mnt_min2', 240) + 30) AS mins_to_dropoff
  FROM mds.trips_canon tc
  WHERE mds.rand01(tc.trip_id || '|mnt') < 0.05
)
INSERT INTO mds.status_changes (vehicle_id, event_time, event_type, area_id, trip_id, note)
SELECT
  COALESCE(NULLIF(m.device_id,''), 'V' || lpad(mds.hash_mod(m.trip_id, 5000)::text, 5, '0')),
  m.ended_at + make_interval(mins => m.mins_to_pickup),
  'maintenance_pickup',
  COALESCE(m.end_area_id, m.start_area_id),
  m.trip_id,
  'Scheduled/unscheduled maintenance'
FROM m;

-- maintenance — dropoff
WITH m AS (
  SELECT tc.*,
         (mds.hash_mod(tc.trip_id || '|mnt_min1', 120) + 60) AS mins_to_pickup,
         (mds.hash_mod(tc.trip_id || '|mnt_min2', 240) + 30) AS mins_to_dropoff
  FROM mds.trips_canon tc
  WHERE mds.rand01(tc.trip_id || '|mnt') < 0.05
)
INSERT INTO mds.status_changes (vehicle_id, event_time, event_type, area_id, trip_id, note)
SELECT
  COALESCE(NULLIF(m.device_id,''), 'V' || lpad(mds.hash_mod(m.trip_id, 5000)::text, 5, '0')),
  m.ended_at + make_interval(mins => m.mins_to_pickup + m.mins_to_dropoff),
  'maintenance_dropoff',
  COALESCE(m.end_area_id, m.start_area_id),
  m.trip_id,
  'Scheduled/unscheduled maintenance'
FROM m;

-- incidents (~3%)
TRUNCATE mds.incidents RESTART IDENTITY;
WITH base AS (
  SELECT tc.*,
         LEAST(
           GREATEST(mds.hash_mod(tc.trip_id || '|inc_offset',
                    GREATEST(1, EXTRACT(EPOCH FROM (tc.ended_at - tc.started_at))::int)),1),
           GREATEST(1, EXTRACT(EPOCH FROM (tc.ended_at - tc.started_at))::int) - 1
         ) AS offset_sec
  FROM mds.trips_canon tc
  WHERE mds.rand01(tc.trip_id || '|inc') < 0.03
),
typed AS (
  SELECT b.*,
         CASE mds.hash_mod(b.trip_id || '|inctype', 4)
           WHEN 1 THEN 'crash'
           WHEN 2 THEN 'vandalism'
           WHEN 3 THEN 'illegal_parking'
           ELSE 'battery_failure'
         END AS incident_type,
         (mds.hash_mod(b.trip_id || '|sev', 3)) AS severity
  FROM base b
)
INSERT INTO mds.incidents (vehicle_id, occurred_at, incident_type, severity, area_id, trip_id, resolved_at, resolution)
SELECT
  COALESCE(NULLIF(t.device_id,''), 'V' || lpad(mds.hash_mod(t.trip_id, 5000)::text, 5, '0')),
  t.started_at + make_interval(secs => t.offset_sec),
  t.incident_type, t.severity,
  COALESCE(t.start_area_id, t.end_area_id),
  t.trip_id,
  t.ended_at + make_interval(mins => mds.hash_mod(t.trip_id || '|resolve', 1440)),
  CASE mds.hash_mod(t.trip_id || '|res', 3)
    WHEN 1 THEN 'Resolved on site'
    WHEN 2 THEN 'Vehicle swapped'
    ELSE 'No issue found'
  END
FROM typed t;
