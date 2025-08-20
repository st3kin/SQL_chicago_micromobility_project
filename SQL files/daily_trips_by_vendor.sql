-- Q1: Daily trips and each vendor's share
-- Output columns: day, vendor, trips, total_trips, vendor_share

WITH by_vendor AS (
  SELECT
    date_trunc('day', started_at)::DATE AS day,
    vendor,
    COUNT(*) AS trips
  FROM mart.trips_canon
  WHERE started_at IS NOT NULL
    AND vendor IS NOT NULL
    -- Optional date filter:
    -- AND started_at >= DATE '2024-07-01'
    -- AND started_at <  DATE '2024-10-01'
  GROUP BY 1,2
),
daily_total AS (
  SELECT 
    day, 
    SUM(trips) AS total_trips
  FROM by_vendor
  GROUP BY 1
)
SELECT
  v.day,
  v.vendor,
  v.trips,
  t.total_trips,
  ROUND((v.trips::NUMERIC / NULLIF(t.total_trips,0))::NUMERIC, 4) AS vendor_share
FROM by_vendor AS v
JOIN daily_total AS t USING (day)
ORDER BY 
    v.day, 
    v.vendor;
