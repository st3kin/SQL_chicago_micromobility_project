-- Qm4: Breakdown of incidents by type and day
-- Uses occurred_at instead of timestamp

WITH inc AS (
  SELECT
    date_trunc('day', occurred_at)::date AS day,
    incident_type
  FROM mds.incidents
)
SELECT
  day,
  incident_type,
  COUNT(*) AS incidents
FROM inc
GROUP BY day, incident_type
ORDER BY day, incident_type;
