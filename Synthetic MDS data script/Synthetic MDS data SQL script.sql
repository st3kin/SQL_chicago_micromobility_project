DO $$
DECLARE
  t regclass := 'e_scooter_trips'::regclass;

  c_trip_id  text;
  c_start_ts text; c_end_ts text;
  c_start_area_num  text; c_end_area_num  text;
  c_start_area_name text; c_end_area_name text;
  c_device_id text; c_vendor text;

  trip_id_expr text; start_ts_expr text; end_ts_expr text;
  start_area_num_expr text; end_area_num_expr text;
  start_area_name_expr text; end_area_name_expr text;
  device_id_expr text; vendor_expr text;

  start_area_id_expr text; end_area_id_expr text;
  sql text;
BEGIN
  -- Autodetect columns
  c_trip_id  := mds.pick_col(t, ARRAY['trip_id','id','rental_id']);
  c_start_ts := mds.pick_col(t, ARRAY['started_at','start_time','start_datetime','trip_start_time','start_time_utc']);
  c_end_ts   := mds.pick_col(t, ARRAY['ended_at','end_time','end_datetime','trip_end_time','end_time_utc']);
  c_start_area_num  := mds.pick_col(t, ARRAY['start_community_area_number','start_area_id','start_community_area']);
  c_end_area_num    := mds.pick_col(t, ARRAY['end_community_area_number','end_area_id','end_community_area']);
  c_start_area_name := mds.pick_col(t, ARRAY['start_community_area_name','start_area_name']);
  c_end_area_name   := mds.pick_col(t, ARRAY['end_community_area_name','end_area_name']);
  c_device_id := mds.pick_col(t, ARRAY['device_id','vehicle_id','bike_id','asset_id']);
  c_vendor    := mds.pick_col(t, ARRAY['vendor','company','provider','provider_name']);

  IF c_start_ts IS NULL OR c_end_ts IS NULL THEN
    RAISE EXCEPTION 'Could not find start/end time columns on %. Expected started_at/ended_at or start_time/end_time.', t::text;
  END IF;

  -- NULL-safe formatted pieces
  trip_id_expr := CASE
    WHEN c_trip_id IS NOT NULL THEN format('%I::text', c_trip_id)
    ELSE format(
      'md5(coalesce(%1$I::text,'''')||''|''||coalesce(%2$I::text,'''')||''|''||coalesce(%1$I::text,''''))',
      c_start_ts, c_end_ts
    )
  END;

  start_ts_expr := format('%I::timestamp', c_start_ts);
  end_ts_expr   := format('%I::timestamp', c_end_ts);

  start_area_num_expr  := CASE WHEN c_start_area_num  IS NULL THEN 'NULL' ELSE format('%I', c_start_area_num)  END;
  end_area_num_expr    := CASE WHEN c_end_area_num    IS NULL THEN 'NULL' ELSE format('%I', c_end_area_num)    END;
  start_area_name_expr := CASE WHEN c_start_area_name IS NULL THEN 'NULL' ELSE format('%I::text', c_start_area_name) END;
  end_area_name_expr   := CASE WHEN c_end_area_name   IS NULL THEN 'NULL' ELSE format('%I::text', c_end_area_name)   END;
  device_id_expr := CASE WHEN c_device_id IS NULL THEN 'NULL' ELSE format('%I::text', c_device_id) END;
  vendor_expr    := CASE WHEN c_vendor    IS NULL THEN 'NULL' ELSE format('%I::text', c_vendor)    END;

  -- Use helper mds.int_from_text(); fallback to hashing area names when numbers missing
  start_area_id_expr := format(
    'COALESCE(mds.int_from_text((%1$s)::text),
             CASE WHEN (%2$s) IS NOT NULL THEN mds.hash_mod((%2$s)::text, 10000) END)',
    start_area_num_expr, start_area_name_expr
  );

  end_area_id_expr := format(
    'COALESCE(mds.int_from_text((%1$s)::text),
             CASE WHEN (%2$s) IS NOT NULL THEN mds.hash_mod((%2$s)::text, 10000) END)',
    end_area_num_expr, end_area_name_expr
  );

  sql := format($f$
    CREATE OR REPLACE VIEW mds.trips_canon AS
    SELECT
      %1$s AS trip_id,
      %2$s AS started_at,
      %3$s AS ended_at,
      %4$s AS start_area_id,
      %5$s AS end_area_id,
      %6$s AS start_area_name,
      %7$s AS end_area_name,
      %8$s AS device_id,
      %9$s AS vendor
    FROM %10$s
    WHERE %2$s IS NOT NULL AND %3$s IS NOT NULL
  $f$,
    trip_id_expr, start_ts_expr, end_ts_expr,
    start_area_id_expr, end_area_id_expr,
    start_area_name_expr, end_area_name_expr,
    device_id_expr, vendor_expr, t::text
  );

  EXECUTE sql;
END
$$;
