-- Qm2: Average downtime per vehicle after maintenance
-- Logic: find pairs of maintenance_pickup → next maintenance_dropoff for each vehicle

WITH ordered AS (
  SELECT
    vehicle_id,
    event_time,
    event_type,
    LEAD(event_time) OVER (PARTITION BY vehicle_id ORDER BY event_time) AS next_time,
    LEAD(event_type) OVER (PARTITION BY vehicle_id ORDER BY event_time) AS next_type
  FROM mds.status_changes
  WHERE event_type IN ('maintenance_pickup', 'maintenance_dropoff')
),
pairs AS (
  SELECT
    vehicle_id,
    event_time AS pickup_time,
    next_time   AS dropoff_time,
    EXTRACT(EPOCH FROM (next_time - event_time))/3600.0 AS downtime_hours
  FROM ordered
  WHERE event_type = 'maintenance_pickup'
    AND next_type  = 'maintenance_dropoff'
    AND next_time IS NOT NULL
    AND next_time > event_time
)
SELECT
  date_trunc('day', pickup_time)::date AS day,
  COUNT(*)                             AS maint_pairs,
  ROUND(AVG(downtime_hours), 2)        AS avg_downtime_h,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY downtime_hours) AS p50_downtime_h,
  PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY downtime_hours) AS p90_downtime_h
FROM pairs
GROUP BY day
ORDER BY day;
