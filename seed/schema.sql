-- OptiQuery seed schema: a deliberately under-indexed e-commerce database.
--
-- THERE ARE NO SECONDARY INDEXES IN THIS FILE. That is not an oversight, it is
-- the entire point: every table carries its PRIMARY KEY and nothing else, so
-- the four queries in slow_queries.sql are forced into sequential scans, hash
-- joins over whole tables, and unusable-predicate plans.
--
-- There are also no FOREIGN KEY constraints. Two reasons:
--   1. FK validation during a 3M-row COPY is the single slowest part of the
--      load and would blow the 3-minute budget.
--   2. A FOREIGN KEY does not create an index on the referencing column in
--      Postgres. Adding them would change nothing about the plans below, so
--      they would buy realism we do not need at a cost we cannot afford.
-- The relationships are documented per column instead.

DROP TABLE IF EXISTS order_items CASCADE;
DROP TABLE IF EXISTS reviews     CASCADE;
DROP TABLE IF EXISTS orders      CASCADE;
DROP TABLE IF EXISTS products    CASCADE;
DROP TABLE IF EXISTS users       CASCADE;

-- 200,000 rows
CREATE TABLE users (
    id             bigint         PRIMARY KEY,
    -- Stored in MIXED CASE on purpose ("Daniel.Reyes137421@Example.com").
    -- This is what makes `WHERE lower(email) = '...'` (slow query 3) a real
    -- problem rather than a cosmetic one: a plain btree on email cannot answer
    -- a lowercased literal, so a rewrite that drops the lower() call silently
    -- returns zero rows. The verifier's checksum is what catches that.
    email          text           NOT NULL,
    full_name      text           NOT NULL,
    -- Heavily skewed. 'IS' is ~0.2% of users and drives slow query 2.
    country        char(2)        NOT NULL,
    city           text           NOT NULL,
    signup_ts      timestamptz    NOT NULL,
    loyalty_tier   text           NOT NULL,
    lifetime_value numeric(12,2)  NOT NULL,
    is_active      boolean        NOT NULL
);

-- 50,000 rows
CREATE TABLE products (
    id         bigint        PRIMARY KEY,
    sku        text          NOT NULL,   -- 'SKU-%07d', derived from id
    name       text          NOT NULL,
    category   text          NOT NULL,
    brand      text          NOT NULL,
    price      numeric(10,2) NOT NULL,
    stock_qty  integer       NOT NULL,
    created_at timestamptz   NOT NULL,
    avg_rating numeric(3,2)
);

-- 1,000,000 rows
CREATE TABLE orders (
    id               bigint        PRIMARY KEY,
    -- logical FK -> users(id). UNINDEXED: this is slow query 2.
    user_id          bigint        NOT NULL,
    status           text          NOT NULL,
    created_at       timestamptz   NOT NULL,
    total_amount     numeric(12,2) NOT NULL,
    -- Denormalised copy of users.email as it was at order time, mixed case.
    -- Drives slow query 3 (non-sargable lower()).
    email_snapshot   text          NOT NULL,
    -- NULL until the order ships (~35% NULL). Drives slow query 4.
    tracking_number  text,
    shipping_country char(2)       NOT NULL,
    payment_method   text          NOT NULL,
    coupon_code      text
);

-- 3,000,000 rows. Physically clustered by order_id because items are written
-- with their order, which is what a real system does.
CREATE TABLE order_items (
    id               bigint        PRIMARY KEY,
    -- logical FK -> orders(id). UNINDEXED: this is the second half of slow query 2.
    order_id         bigint        NOT NULL,
    -- logical FK -> products(id).
    product_id       bigint        NOT NULL,
    -- Denormalised copy of products.sku. ~50,000 distinct values over 3M rows,
    -- i.e. high cardinality and highly selective. UNINDEXED: slow query 1.
    sku              text          NOT NULL,
    quantity         integer       NOT NULL,
    unit_price       numeric(10,2) NOT NULL,
    discount_pct     numeric(5,2)  NOT NULL,
    warehouse_code   text          NOT NULL,
    -- Free-text operational field. Present to give the table a realistic row
    -- width (~180 bytes); a narrow table would make a 3M-row sequential scan
    -- too cheap to be an interesting optimisation target.
    fulfillment_note text          NOT NULL
);

-- 500,000 rows
CREATE TABLE reviews (
    id            bigint      PRIMARY KEY,
    product_id    bigint      NOT NULL,  -- logical FK -> products(id)
    user_id       bigint      NOT NULL,  -- logical FK -> users(id)
    rating        smallint    NOT NULL,
    title         text        NOT NULL,
    body          text        NOT NULL,
    created_at    timestamptz NOT NULL,
    helpful_votes integer     NOT NULL
);
