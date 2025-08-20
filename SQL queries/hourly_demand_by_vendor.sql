-- Q3: Hour-of-day demand profile per vendor (with normalized shares)
-- Output (long format): day, vendor, hod, trips, vendor_day_total, share_within_vendor_day, share_within_day
-- Notes:
--   • Uses same trip-quality filters as Q2 to keep datasets consistent.
--   • share_within_vendor_day = each hour’s share of that vendor’s daily trips.
--   • share_within_day        = each vendor-hour’s share of the whole market that day.

WITH base AS (
  SELECT
    date_trunc('day', started_at)::DATE AS day,
    EXTRACT(hour FROM started_at)::INT  AS hour,
    vendor,
    duration_s,
    distance_m,
    (distance_m/1000.0) / (duration_s/3600.0) AS speed_kmh
  FROM mart.trips_canon
  WHERE duration_s IS NOT NULL
    AND duration_s > 60
    AND distance_m IS NOT NULL
    AND distance_m > 0
    AND distance_m/1000.0 <= 20               -- drop extreme distances
    AND (distance_m/1000.0)/(duration_s/3600.0) BETWEEN 5 AND 30  -- realistic speeds
),
counts AS (
  SELECT
    day, 
    hour, 
    vendor,
    COUNT(*) AS trips
  FROM base
  GROUP BY 1,2,3
),
vendor_day_tot AS (
  SELECT 
    day, 
    vendor, 
    SUM(trips) AS vendor_day_total
  FROM counts
  GROUP BY 1,2
),
day_tot AS (
  SELECT 
    day, 
    SUM(trips) AS day_total
  FROM counts
  GROUP BY 1
)
SELECT
  c.day,
  c.vendor,
  c.hour,
  c.trips,
  v.vendor_day_total,
  d.day_total,
  ROUND( (c.trips::numeric / NULLIF(v.vendor_day_total,0)), 4) AS share_within_vendor_day,
  ROUND( (c.trips::numeric / NULLIF(d.day_total,0)),    4)     AS share_within_day
FROM counts AS c
JOIN vendor_day_tot v USING (day, vendor)
JOIN day_tot d        USING (day)
ORDER BY 
    c.day, 
    c.vendor, 
    c.hour;
