-- Q2 (filtered): Average duration & distance by vendor/day
-- Filters:
--   - 0 < distance ≤ 20 km (drop distance outliers)
--   - duration > 60 s (drop near-zero records)
--   - speed between 3 and 35 km/h (drop parked/paused or crazy-fast trips)

WITH trips AS (
  SELECT
    vendor,
    date_trunc('day', started_at)::date AS day,
    duration_s,
    (distance_m / 1000.0)               AS distance_km,
    (distance_m / 1000.0) / (duration_s / 3600.0) AS speed_kmh
  FROM mart.trips_canon
  WHERE duration_s IS NOT NULL
    AND duration_s > 60
    AND distance_m IS NOT NULL
    AND distance_m > 0
    AND (distance_m / 1000.0) <= 20
),
clean AS (
  SELECT *
  FROM trips
  WHERE speed_kmh BETWEEN 3 AND 35
)
SELECT
  vendor,
  day,
  COUNT(*)                                        AS trips,
  ROUND(AVG(duration_s)/60.0, 2)                  AS avg_duration_min,
  ROUND(AVG(distance_km), 2)                      AS avg_distance_km,
  PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY distance_km) AS p50_km,
  PERCENTILE_CONT(0.9) WITHIN GROUP (ORDER BY distance_km) AS p90_km
FROM clean
GROUP BY vendor, day
ORDER BY day, vendor;
