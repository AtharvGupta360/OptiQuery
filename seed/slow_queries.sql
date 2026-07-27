-- OptiQuery seed workload: four slow queries, four distinct failure modes.
--
-- Every literal below was read back out of the loaded data by
-- seed/generate_data.py (see seed/query_params.json). The generator is
-- deterministic, so these values stay valid across reloads.
--
-- Blocks are delimited by `-- name:` markers; seed/measure_baseline.py and the
-- Makefile parse on those markers, so keep the format.
--
-- Two things worth stating up front, because they shape how these queries look:
--
--   1. The databases run with max_parallel_workers_per_gather=0 (see
--      docker-compose.yml). These are single-threaded numbers. The same setting
--      applies to every optimised variant the verifier benchmarks later, so no
--      before/after comparison is distorted by it.
--
--   2. Some of these queries join a second table that is not itself the
--      problem. That is deliberate. On a 16-core machine a bare single-table
--      filter over 3M rows returns in ~380ms -- too fast to be worth
--      optimising, and too fast for a 20% improvement threshold to mean
--      anything. Real slow queries are reports, not bare filters. Each query
--      below is still *dominated* by one named failure mode, and the EXPLAIN
--      output printed by seed/measure_baseline.py shows which node owns the
--      time.


-- name: q1_seq_scan_high_cardinality
-- FAILURE MODE 1: filter on an unindexed high-cardinality column.
--
-- order_items.sku has ~50,000 distinct values across 3,000,000 rows, and this
-- literal matches 67 of them -- 0.002% selectivity, the textbook case for an
-- index. There is no index, so Postgres reads all 3,000,000 rows and ~1GB of
-- heap to return a few dozen.
--
-- It pays that cost TWICE: the subquery establishing the SKU's average selling
-- price is a second, independent sequential scan of the same table with the
-- same unusable predicate. One missing index, two full scans. Single table, no
-- joins -- this is the purest of the four.
--
-- Expected fix: CREATE INDEX ON order_items (sku), which turns both scans into
-- index scans.
SELECT oi.id,
       oi.order_id,
       oi.quantity,
       oi.unit_price,
       oi.warehouse_code
FROM order_items oi
WHERE oi.sku = 'SKU-0042137'
  AND oi.unit_price > (
        SELECT avg(x.unit_price)
        FROM order_items x
        WHERE x.sku = 'SKU-0042137'
      )
ORDER BY oi.id;


-- name: q2_unindexed_fk_join
-- FAILURE MODE 2: join on an unindexed foreign key.
--
-- 395 of 200,000 users are in 'IS'. Their orders are ~0.2% of the orders table.
-- With no index on orders.user_id the planner has no way to fetch just those
-- rows, so it hash-joins the whole 1,000,000-row orders table -- and then the
-- whole 3,000,000-row order_items table on top of that -- to produce 25 output
-- rows.
--
-- Expected fix: CREATE INDEX ON orders (user_id) and
--               CREATE INDEX ON order_items (order_id), turning both hash
--               joins into nested loops over a few thousand index lookups.
SELECT u.id,
       u.email,
       u.city,
       count(DISTINCT o.id)             AS order_count,
       sum(oi.quantity * oi.unit_price) AS gross_revenue
FROM users u
JOIN orders o       ON o.user_id  = u.id
JOIN order_items oi ON oi.order_id = o.id
WHERE u.country = 'IS'
GROUP BY u.id, u.email, u.city
ORDER BY gross_revenue DESC, u.id
LIMIT 25;


-- name: q3_non_sargable_lower
-- FAILURE MODE 3: non-sargable predicate -- a function call wrapped around the
-- column being filtered.
--
-- orders.email_snapshot is stored mixed-case. Wrapping it in lower() makes the
-- predicate unusable by any plain btree on email_snapshot: the index is sorted
-- on 'Paulo.Lindqvist...', the query asks for 'paulo.lindqvist...'. So every
-- one of the 1,000,000 rows is read and lower() is evaluated on each.
--
-- This one is a trap for the agent, deliberately. The obvious-looking fix --
-- index email_snapshot directly and rewrite the predicate to drop the lower()
-- call -- runs in under a millisecond and returns ZERO rows instead of 15.
-- Only the result checksum catches that. A runtime comparison on its own would
-- score it as a 99.9% improvement and recommend it.
--
-- Expected fix: CREATE INDEX ON orders (lower(email_snapshot)).
SELECT o.id,
       o.created_at,
       o.status,
       o.total_amount,
       count(oi.id)                     AS line_items,
       sum(oi.quantity * oi.unit_price) AS items_total
FROM orders o
JOIN order_items oi ON oi.order_id = o.id
WHERE lower(o.email_snapshot) = 'paulo.lindqvist138671@example.com'
GROUP BY o.id, o.created_at, o.status, o.total_amount
ORDER BY o.created_at DESC, o.id;


-- name: q4_or_across_columns
-- FAILURE MODE 4: OR spanning columns in two different tables.
--
-- A single-table OR is not actually a problem in Postgres -- given indexes on
-- both columns the planner builds a BitmapOr and is done. An OR whose arms sit
-- on *different* tables is the one that genuinely blocks index use: neither arm
-- can be evaluated before the join, so the planner must materialise the full
-- 1,000,000-row join and filter afterwards. Adding indexes changes nothing;
-- this failure mode can only be fixed by rewriting the query.
--
-- Expected fix: rewrite as UNION ALL so each arm becomes independently
-- sargable, with an anti-predicate on the second arm so rows satisfying BOTH
-- conditions are not emitted twice. Getting that anti-predicate wrong changes
-- the result set -- which is exactly what the checksum exists to catch.
--
--   SELECT ... WHERE u.email = 'Kiran.Dubois89332@Example.com'
--   UNION ALL
--   SELECT ... WHERE o.tracking_number = 'TRK-911393315'
--                AND u.email IS DISTINCT FROM 'Kiran.Dubois89332@Example.com'
SELECT o.id,
       o.created_at,
       o.status,
       o.total_amount,
       u.email,
       oi.sku,
       oi.quantity
FROM orders o
JOIN users u        ON u.id = o.user_id
JOIN order_items oi ON oi.order_id = o.id
WHERE u.email = 'Kiran.Dubois89332@Example.com'
   OR o.tracking_number = 'TRK-911393315'
ORDER BY o.id, oi.sku;
