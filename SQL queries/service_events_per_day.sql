-- Qm1: Count of scooters going into maintenance per day
-- We treat "maintenance_pickup" as out-of-service.

WITH changes AS (
  SELECT
    date_trunc('day', event_time)::date AS day,
    vehicle_id
  FROM mds.status_changes
  WHERE event_type = 'maintenance_pickup'
)
SELECT
  day,
  COUNT(*) AS maintenance_events,
  COUNT(DISTINCT vehicle_id) AS unique_vehicles
FROM changes
GROUP BY day
ORDER BY day;
