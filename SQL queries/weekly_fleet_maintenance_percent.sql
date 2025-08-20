-- Qm3: % of fleet that required maintenance each week
-- Assumes mds.vehicles contains all active fleet vehicles

WITH weekly_events AS (
  SELECT
    date_trunc('week', event_time)::date AS week_start,
    vehicle_id
  FROM mds.status_changes
  WHERE event_type = 'maintenance_pickup'
  GROUP BY 1, vehicle_id
),
fleet AS (
  SELECT COUNT(DISTINCT vehicle_id) AS fleet_size
  FROM mds.vehicles
)
SELECT
  w.week_start,
  COUNT(DISTINCT w.vehicle_id) AS vehicles_in_maintenance,
  f.fleet_size,
  ROUND( (COUNT(DISTINCT w.vehicle_id)::numeric / f.fleet_size) * 100, 2) AS pct_fleet_maintenance
FROM weekly_events w
CROSS JOIN fleet f
GROUP BY w.week_start, f.fleet_size
ORDER BY w.week_start;
