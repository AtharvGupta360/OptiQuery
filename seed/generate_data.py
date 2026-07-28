"""Deterministic bulk data generator for the OptiQuery seed database.

Design constraints, in priority order:

1.  **COPY FROM STDIN, never INSERT.** Row-by-row INSERT of 4.75M rows takes
    tens of minutes. Streaming pre-formatted text into COPY takes seconds.
2.  **Under ~3 minutes for a full load.** Everything below is shaped by that:
    values are drawn from precomputed pools with ``random.choices`` (the fast
    uniform C path) rather than being computed per row, and rows are formatted
    as raw COPY text lines instead of being round-tripped through parameter
    adaptation.
3.  **Deterministic.** Every table is generated from a fixed seed, so the
    literal values baked into ``slow_queries.sql`` stay valid across reloads
    and are identical on ``primary`` and ``shadow``.
4.  **VACUUM ANALYZE afterwards.** COPY leaves the planner with no statistics
    at all, and without statistics every plan -- and therefore every benchmark
    number downstream -- is meaningless. VACUUM (not just ANALYZE) additionally
    sets the visibility map, which COPY leaves cleared.

The generated text never contains a tab, newline, backslash or carriage
return, so no COPY escaping is required. ``_assert_copy_safe`` enforces that
against the source pools at import time rather than trusting the claim.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator, Sequence

import psycopg

SEED_DIR = Path(__file__).resolve().parent
SCHEMA_PATH = SEED_DIR / "schema.sql"
PARAMS_PATH = SEED_DIR / "query_params.json"

# ---------------------------------------------------------------------------
# Row counts. Fixed by the project spec; overridable only to make local
# smoke-testing of this file cheap (OPTIQUERY_SCALE=0.01 -> 1% of every table).
# ---------------------------------------------------------------------------
# `or` rather than a get() default: env_file passes declared-but-blank keys
# through as empty strings, and float("") raises.
SCALE = float(os.environ.get("OPTIQUERY_SCALE") or "1.0")

N_USERS = int(200_000 * SCALE)
N_PRODUCTS = int(50_000 * SCALE)
N_ORDERS = int(1_000_000 * SCALE)
N_ORDER_ITEMS = int(3_000_000 * SCALE)
N_REVIEWS = int(500_000 * SCALE)

CHUNK = 50_000

# Seeds are per-table so tables stay reproducible independently of each other.
SEED_USERS = 1_000_003
SEED_PRODUCTS = 1_000_033
SEED_ORDERS = 1_000_037
SEED_ITEMS = 1_000_039
SEED_REVIEWS = 1_000_081

# Rows whose values get baked into slow_queries.sql as literals.
PARAM_USER_Q3 = min(137_421, N_USERS)  # email_snapshot target for query 3
PARAM_USER_Q4 = min(88_123, N_USERS)  # email target for query 4
PARAM_ORDER_Q4 = min(731_905, N_ORDERS)  # tracking_number target for query 4
PARAM_PRODUCT_Q1 = min(42_137, N_PRODUCTS)  # sku target for query 1

RARE_COUNTRY = "IS"  # ~0.2% of users; drives query 2

# ---------------------------------------------------------------------------
# Value pools
# ---------------------------------------------------------------------------

FIRST_NAMES: Sequence[str] = (
    "Aaron Adele Alba Alexei Amara Anders Anika Arjun Beatriz Bjorn Camila Cato "
    "Chidi Clara Damian Daniel Dilara Eero Elena Elias Emeka Esther Fatima Felix "
    "Freya Gabriel Giulia Hannah Haruto Hugo Ingrid Isabel Ivan Jasmin Joachim "
    "Jonas Julia Kaito Karim Katrin Kiran Lars Laura Leila Liam Linnea Lucas "
    "Magnus Maja Marco Mateo Mei Mikkel Nadia Niamh Noor Olof Omar Paulo Petra "
    "Priya Rafael Rania Ravi Rosa Sanne Sofia Stefan Tariq Tomas Ulrike Valeria "
    "Viktor Wei Yara Yusuf Zainab Zoltan"
).split()

LAST_NAMES: Sequence[str] = (
    "Abbott Adeyemi Almeida Andersen Ashford Bakker Barros Beck Berger Bhatt "
    "Blomqvist Carvalho Castillo Chen Conti Dahl Delacroix Dubois Eriksen Farkas "
    "Fernandez Fischer Gallagher Ghosh Grimaldi Haddad Hansen Hoffmann Ibrahim "
    "Iversen Jansen Kaminski Kato Keller Khan Kovac Lambert Larsen Lindqvist "
    "Lopez Maier Marchetti Mensah Moreau Nakamura Navarro Nilsson Novak Okafor "
    "Oliveira Ortega Pavlov Pereira Petrov Quinn Rasmussen Reyes Ricci Romano "
    "Rossi Saito Sandoval Schneider Silva Sorensen Stark Suzuki Tanaka Thomsen "
    "Torres Vargas Virtanen Wagner Walsh Weber Yamada Zielinski"
).split()

CITIES: Sequence[str] = (
    "Amsterdam Antwerp Athens Auckland Austin Barcelona Bergen Berlin Bilbao "
    "Bologna Bordeaux Bristol Brno Bruges Budapest Chicago Cologne Copenhagen "
    "Cork Denver Dresden Dublin Edinburgh Eindhoven Florence Frankfurt Geneva "
    "Ghent Glasgow Gothenburg Graz Hamburg Helsinki Innsbruck Istanbul Kyoto "
    "Leeds Leipzig Lisbon Ljubljana Lyon Madrid Malmo Manchester Marseille "
    "Melbourne Milan Montreal Munich Nantes Naples Nice Osaka Oslo Ottawa Porto "
    "Prague Reykjavik Riga Rotterdam Salzburg Seattle Seville Sofia Stockholm "
    "Stuttgart Tallinn Tampere Toronto Toulouse Trieste Turin Utrecht Valencia "
    "Vancouver Venice Verona Vienna Vilnius Warsaw Zagreb Zurich"
).split()

# 1000-entry expansion so the (fast, unweighted) random.choices path produces a
# realistically skewed country distribution. 'IS' appears twice => 0.2%.
_COUNTRY_WEIGHTS: dict[str, int] = {
    "US": 240, "DE": 120, "GB": 95, "FR": 80, "IN": 70, "BR": 55, "CA": 48,
    "NL": 42, "IT": 40, "ES": 38, "AU": 30, "SE": 26, "PL": 24, "JP": 22,
    "MX": 20, "NO": 14, "DK": 13, "FI": 12, "IE": 11, "PT": 10, "AT": 9,
    "CH": 8, "BE": 7, "CZ": 6, "GR": 5, "NZ": 4, "SG": 3, "EE": 3, "LV": 2,
    "IS": 2,
}
COUNTRY_POOL: list[str] = [c for c, w in _COUNTRY_WEIGHTS.items() for _ in range(w)]

LOYALTY_TIERS: Sequence[str] = ("bronze",) * 55 + ("silver",) * 28 + ("gold",) * 13 + ("platinum",) * 4

PRODUCT_CATEGORIES: Sequence[str] = (
    "audio books camping cookware cycling electronics footwear furniture "
    "gardening grocery haircare hardware kitchen lighting luggage networking "
    "outdoors pet-supplies photography power-tools skincare sportswear "
    "stationery storage toys watches"
).split()

BRANDS: Sequence[str] = (
    "Alturia Bexon Caldera Dovetail Ellery Fairhaven Glimmer Halcyon Ironwood "
    "Junipero Kestrel Lumen Meridian Northvale Oakhurst Pinnacle Quarry Ridgeway "
    "Solstice Thornbury Umbra Vantage Westmark Yarrow Zephyr"
).split()

PRODUCT_NOUNS: Sequence[str] = (
    "Adapter Backpack Blender Bottle Bracket Cable Chair Charger Clamp Cooler "
    "Desk Dock Drill Duffel Fan Filter Grinder Headset Hub Jacket Kettle Lamp "
    "Lantern Mat Monitor Mount Mug Panel Pump Rack Router Sensor Shelf Speaker "
    "Stand Stool Switch Tripod Trolley Vacuum Wallet Wrench"
).split()

PRODUCT_ADJECTIVES: Sequence[str] = (
    "Alpine Aero Basalt Carbon Classic Compact Coastal Core Delta Dual Eco Elite "
    "Everyday Field Flux Granite Harbor Heritage Lite Matte Nimbus Nordic Pro "
    "Quartz Rapid Slate Studio Summit Trail Ultra Urban Vertex"
).split()

ORDER_STATUSES: Sequence[str] = (
    ("delivered",) * 58 + ("shipped",) * 17 + ("processing",) * 12
    + ("pending",) * 8 + ("cancelled",) * 4 + ("refunded",) * 1
)

PAYMENT_METHODS: Sequence[str] = (
    ("card",) * 62 + ("paypal",) * 18 + ("bank_transfer",) * 9
    + ("apple_pay",) * 7 + ("gift_card",) * 4
)

COUPON_CODES: Sequence[str] = (
    "WELCOME10 SPRING15 SUMMER20 AUTUMN12 WINTER25 FREESHIP LOYALTY5 BUNDLE30 "
    "FLASH40 REFER15"
).split()

WAREHOUSE_CODES: Sequence[str] = [
    f"WH-{region}-{n:02d}" for region in ("EU", "NA", "APAC") for n in range(1, 13)
]

# Fragments joined 3-5 at a time to build a ~200 character fulfillment note or
# review body. Keeping the pool small keeps generation fast; the resulting row
# width is what matters for the sequential-scan cost, not the vocabulary.
NOTE_FRAGMENTS: Sequence[str] = (
    "packed in recycled outer carton",
    "fragile item, corner protectors applied",
    "consolidated with adjacent line item",
    "customer requested no invoice in box",
    "gift wrap applied at pick station",
    "split shipment, second parcel to follow",
    "weight verified at outbound scale",
    "hazmat screening cleared",
    "oversize surcharge applied by carrier",
    "picked from overflow bay",
    "serial number recorded at dispatch",
    "battery declaration attached",
    "temperature sensitive, insulated liner",
    "signature required on delivery",
    "address corrected by carrier lookup",
    "repack required after damaged inner box",
    "expedited to meet cutoff window",
    "held briefly for fraud review",
    "label reprinted after scanner rejection",
    "returned to stock and re-picked",
)

REVIEW_TITLES: Sequence[str] = (
    "Exactly what I needed", "Good but not great", "Would buy again",
    "Disappointed with the finish", "Solid value for the price",
    "Arrived faster than expected", "Not as described", "Excellent build quality",
    "Works well after a firmware update", "Broke within a month",
    "Better than the previous model", "Does the job, nothing more",
    "Great for daily use", "Overpriced for what it is", "Very happy with this",
    "Mixed feelings", "Perfect fit", "Instructions were unclear",
    "Sturdier than it looks", "Second one I have bought",
)

REVIEW_FRAGMENTS: Sequence[str] = (
    "the finish is cleaner than the photos suggest",
    "setup took under ten minutes with no tools",
    "it has handled daily use for three weeks without issue",
    "the packaging was minimal which I appreciated",
    "the fit is slightly tighter than the size chart implies",
    "battery life is roughly what the listing claims",
    "the included cable is too short to be useful",
    "customer support answered within a day",
    "there is a faint rattle at higher settings",
    "the weight distribution feels well judged",
    "I would have preferred a matte surface",
    "assembly instructions skip an obvious step",
    "it replaced a unit that lasted four years",
    "the seal is tight and has not leaked once",
    "colour is a shade darker in person",
    "it is noticeably quieter than my old one",
    "the mounting hardware feels underspecified",
    "value is hard to beat at this price point",
    "the app pairing was unreliable at first",
    "returns were straightforward when I sized up",
)


def _assert_copy_safe(pools: Sequence[Sequence[str]]) -> None:
    """COPY text format is tab/newline/backslash delimited.

    Every string that reaches a COPY line comes from one of these pools (or is
    a formatted number/id). If any pool contained a delimiter the load would
    corrupt silently -- shifted columns, not an error. Check instead of hoping.
    """
    bad = "\t\n\r\\"
    for pool in pools:
        for value in pool:
            if any(ch in value for ch in bad):
                raise ValueError(f"pool value is not COPY-safe: {value!r}")


_assert_copy_safe(
    [
        FIRST_NAMES, LAST_NAMES, CITIES, COUNTRY_POOL, LOYALTY_TIERS,
        PRODUCT_CATEGORIES, BRANDS, PRODUCT_NOUNS, PRODUCT_ADJECTIVES,
        ORDER_STATUSES, PAYMENT_METHODS, COUPON_CODES, WAREHOUSE_CODES,
        NOTE_FRAGMENTS, REVIEW_TITLES, REVIEW_FRAGMENTS,
    ]
)


# ---------------------------------------------------------------------------
# Precomputed timestamp pool
# ---------------------------------------------------------------------------

def _timestamp_pool(seed: int, size: int, start: datetime, days: int) -> list[str]:
    """Pool of pre-rendered UTC timestamp literals.

    Formatting a datetime per row costs more than every other per-row operation
    combined. Sampling from a pool of `size` pre-rendered strings is ~40x
    cheaper and the duplicate timestamps it produces are harmless here.
    """
    rnd = random.Random(seed)
    span = days * 86_400
    return [
        (start + timedelta(seconds=rnd.randrange(span))).strftime("%Y-%m-%d %H:%M:%S+00")
        for _ in range(size)
    ]


EPOCH_START = datetime(2022, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Per-table row generators. Each yields COPY-format text chunks.
# ---------------------------------------------------------------------------

def gen_users(emails_out: list[str]) -> Iterator[str]:
    """users. Also fills `emails_out` (index 0 == user id 1) for reuse by orders."""
    rnd = random.Random(SEED_USERS)
    ts_pool = _timestamp_pool(SEED_USERS + 1, 20_000, EPOCH_START, 1095)
    ltv_pool = [f"{rnd.uniform(0, 18_000):.2f}" for _ in range(20_000)]

    for start in range(1, N_USERS + 1, CHUNK):
        n = min(CHUNK, N_USERS + 1 - start)
        firsts = rnd.choices(FIRST_NAMES, k=n)
        lasts = rnd.choices(LAST_NAMES, k=n)
        countries = rnd.choices(COUNTRY_POOL, k=n)
        cities = rnd.choices(CITIES, k=n)
        tiers = rnd.choices(LOYALTY_TIERS, k=n)
        stamps = rnd.choices(ts_pool, k=n)
        ltvs = rnd.choices(ltv_pool, k=n)
        actives = rnd.choices(("t", "t", "t", "t", "f"), k=n)

        rows: list[str] = []
        append = rows.append
        for j in range(n):
            uid = start + j
            first = firsts[j]
            last = lasts[j]
            # Mixed case on purpose -- see schema.sql.
            email = f"{first}.{last}{uid}@Example.com"
            emails_out.append(email)
            append(
                f"{uid}\t{email}\t{first} {last}\t{countries[j]}\t{cities[j]}\t"
                f"{stamps[j]}\t{tiers[j]}\t{ltvs[j]}\t{actives[j]}\n"
            )
        yield "".join(rows)


def gen_products() -> Iterator[str]:
    rnd = random.Random(SEED_PRODUCTS)
    ts_pool = _timestamp_pool(SEED_PRODUCTS + 1, 10_000, EPOCH_START, 1095)
    price_pool = [f"{rnd.uniform(3, 1400):.2f}" for _ in range(20_000)]
    rating_pool = [f"{rnd.uniform(1, 5):.2f}" for _ in range(2_000)]

    for start in range(1, N_PRODUCTS + 1, CHUNK):
        n = min(CHUNK, N_PRODUCTS + 1 - start)
        adjectives = rnd.choices(PRODUCT_ADJECTIVES, k=n)
        nouns = rnd.choices(PRODUCT_NOUNS, k=n)
        brands = rnd.choices(BRANDS, k=n)
        categories = rnd.choices(PRODUCT_CATEGORIES, k=n)
        prices = rnd.choices(price_pool, k=n)
        stocks = rnd.choices(range(0, 900), k=n)
        stamps = rnd.choices(ts_pool, k=n)
        ratings = rnd.choices(rating_pool, k=n)
        has_rating = rnd.choices((True, True, True, True, True, True, True, True, True, False), k=n)

        rows: list[str] = []
        append = rows.append
        for j in range(n):
            pid = start + j
            rating = ratings[j] if has_rating[j] else r"\N"
            append(
                f"{pid}\tSKU-{pid:07d}\t{brands[j]} {adjectives[j]} {nouns[j]}\t"
                f"{categories[j]}\t{brands[j]}\t{prices[j]}\t{stocks[j]}\t"
                f"{stamps[j]}\t{rating}\n"
            )
        yield "".join(rows)


def gen_orders(emails: list[str]) -> Iterator[str]:
    """orders. email_snapshot is the ordering user's email, verbatim mixed case."""
    rnd = random.Random(SEED_ORDERS)
    ts_pool = _timestamp_pool(SEED_ORDERS + 1, 60_000, EPOCH_START, 1095)
    total_pool = [f"{rnd.uniform(8, 2600):.2f}" for _ in range(40_000)]
    # ~35% of orders have not shipped and carry a NULL tracking number.
    tracking_present = ("t",) * 65 + ("f",) * 35

    for start in range(1, N_ORDERS + 1, CHUNK):
        n = min(CHUNK, N_ORDERS + 1 - start)
        user_ids = rnd.choices(range(1, N_USERS + 1), k=n)
        statuses = rnd.choices(ORDER_STATUSES, k=n)
        stamps = rnd.choices(ts_pool, k=n)
        totals = rnd.choices(total_pool, k=n)
        countries = rnd.choices(COUNTRY_POOL, k=n)
        payments = rnd.choices(PAYMENT_METHODS, k=n)
        coupons = rnd.choices(COUPON_CODES, k=n)
        has_coupon = rnd.choices((True, False, False, False), k=n)
        has_tracking = rnd.choices(tracking_present, k=n)
        track_nums = rnd.choices(range(100_000_000, 999_999_999), k=n)

        rows: list[str] = []
        append = rows.append
        for j in range(n):
            oid = start + j
            uid = user_ids[j]
            email = emails[uid - 1]
            tracking = f"TRK-{track_nums[j]}" if has_tracking[j] == "t" else r"\N"
            coupon = coupons[j] if has_coupon[j] else r"\N"
            append(
                f"{oid}\t{uid}\t{statuses[j]}\t{stamps[j]}\t{totals[j]}\t{email}\t"
                f"{tracking}\t{countries[j]}\t{payments[j]}\t{coupon}\n"
            )
        yield "".join(rows)


def _item_counts(rnd: random.Random) -> list[int]:
    """Items per order, summing to exactly N_ORDER_ITEMS.

    A repeating pattern with mean 3.0 shuffled across all orders gives an exact
    total without a fixup pass, and keeps items physically clustered by
    order_id -- which is what a real system produces, and what makes an index
    on order_items(order_id) worth proposing.
    """
    pattern = [1, 2, 3, 4, 5, 3, 2, 4, 3, 3]  # sum 30 over 10 orders
    assert sum(pattern) == 30
    reps, remainder = divmod(N_ORDER_ITEMS, 30)
    counts = pattern * reps
    # Absorb any remainder (only non-zero when OPTIQUERY_SCALE is set) on the tail.
    while remainder:
        take = min(remainder, 6)
        counts.append(take)
        remainder -= take
    # Pad/trim to the order count so every order id is covered deterministically.
    if len(counts) < N_ORDERS:
        counts.extend([0] * (N_ORDERS - len(counts)))
    rnd.shuffle(counts)
    return counts


def gen_order_items() -> Iterator[str]:
    rnd = random.Random(SEED_ITEMS)
    price_pool = [f"{rnd.uniform(3, 1400):.2f}" for _ in range(20_000)]
    discount_pool = [f"{rnd.uniform(0, 35):.2f}" for _ in range(5_000)]
    # ~235 characters per note. Row width is the knob that sets how expensive a
    # full sequential scan of this table is, and three of the four seed queries
    # are dominated by exactly that scan. At ~130 chars the scan measured 376ms,
    # which is not slow enough to be worth optimising; at ~235 it measures ~550ms
    # and every seed query clears the 800ms bar with margin. See
    # seed/measure_baseline.py for the numbers.
    note_pool = [
        ", ".join(rnd.sample(NOTE_FRAGMENTS, k=rnd.randint(6, 8)))
        for _ in range(20_000)
    ]

    counts = _item_counts(random.Random(SEED_ITEMS + 7))

    item_id = 0
    rows: list[str] = []
    append = rows.append
    # Draw the random columns in blocks; per-row calls dominate runtime otherwise.
    block_size = CHUNK
    block: list[tuple] = []
    block_pos = 0

    def refill() -> list[tuple]:
        pids = rnd.choices(range(1, N_PRODUCTS + 1), k=block_size)
        qtys = rnd.choices((1, 1, 1, 2, 2, 3, 4, 5), k=block_size)
        prices = rnd.choices(price_pool, k=block_size)
        discounts = rnd.choices(discount_pool, k=block_size)
        warehouses = rnd.choices(WAREHOUSE_CODES, k=block_size)
        notes = rnd.choices(note_pool, k=block_size)
        return list(zip(pids, qtys, prices, discounts, warehouses, notes))

    block = refill()

    for order_idx, count in enumerate(counts, start=1):
        for _ in range(count):
            if block_pos == block_size:
                block = refill()
                block_pos = 0
            pid, qty, price, discount, warehouse, note = block[block_pos]
            block_pos += 1
            item_id += 1
            append(
                f"{item_id}\t{order_idx}\t{pid}\tSKU-{pid:07d}\t{qty}\t{price}\t"
                f"{discount}\t{warehouse}\t{note}\n"
            )
        if len(rows) >= CHUNK:
            yield "".join(rows)
            rows = []
            append = rows.append
    if rows:
        yield "".join(rows)


def gen_reviews() -> Iterator[str]:
    rnd = random.Random(SEED_REVIEWS)
    ts_pool = _timestamp_pool(SEED_REVIEWS + 1, 40_000, EPOCH_START, 1095)
    body_pool = [
        ". ".join(rnd.sample(REVIEW_FRAGMENTS, k=rnd.randint(3, 5))) + "."
        for _ in range(20_000)
    ]

    for start in range(1, N_REVIEWS + 1, CHUNK):
        n = min(CHUNK, N_REVIEWS + 1 - start)
        product_ids = rnd.choices(range(1, N_PRODUCTS + 1), k=n)
        user_ids = rnd.choices(range(1, N_USERS + 1), k=n)
        ratings = rnd.choices((1, 2, 3, 3, 4, 4, 4, 5, 5, 5), k=n)
        titles = rnd.choices(REVIEW_TITLES, k=n)
        bodies = rnd.choices(body_pool, k=n)
        stamps = rnd.choices(ts_pool, k=n)
        votes = rnd.choices(range(0, 400), k=n)

        rows: list[str] = []
        append = rows.append
        for j in range(n):
            rid = start + j
            append(
                f"{rid}\t{product_ids[j]}\t{user_ids[j]}\t{ratings[j]}\t{titles[j]}\t"
                f"{bodies[j]}\t{stamps[j]}\t{votes[j]}\n"
            )
        yield "".join(rows)


# ---------------------------------------------------------------------------
# Load driver
# ---------------------------------------------------------------------------

TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "users": (
        "id", "email", "full_name", "country", "city", "signup_ts",
        "loyalty_tier", "lifetime_value", "is_active",
    ),
    "products": (
        "id", "sku", "name", "category", "brand", "price", "stock_qty",
        "created_at", "avg_rating",
    ),
    "orders": (
        "id", "user_id", "status", "created_at", "total_amount", "email_snapshot",
        "tracking_number", "shipping_country", "payment_method", "coupon_code",
    ),
    "order_items": (
        "id", "order_id", "product_id", "sku", "quantity", "unit_price",
        "discount_pct", "warehouse_code", "fulfillment_note",
    ),
    "reviews": (
        "id", "product_id", "user_id", "rating", "title", "body", "created_at",
        "helpful_votes",
    ),
}


def _log(target: str, message: str) -> None:
    print(f"[{target:<7}] {message}", flush=True)


def copy_table(
    conn: psycopg.Connection,
    target: str,
    table: str,
    chunks: Iterator[str],
    expected_rows: int,
) -> float:
    columns = ", ".join(TABLE_COLUMNS[table])
    started = time.perf_counter()
    with conn.cursor() as cur:
        with cur.copy(f"COPY {table} ({columns}) FROM STDIN") as copy:
            for chunk in chunks:
                copy.write(chunk)
    conn.commit()

    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")
        row = cur.fetchone()
        assert row is not None
        actual = row[0]
    if actual != expected_rows:
        raise RuntimeError(
            f"{table}: expected {expected_rows} rows, loaded {actual}. "
            "The generator and the row-count constants have diverged."
        )

    elapsed = time.perf_counter() - started
    _log(target, f"COPY {table:<12} {actual:>9,} rows in {elapsed:6.1f}s")
    return elapsed


def capture_query_params(conn: psycopg.Connection) -> dict[str, str]:
    """Read the literals that slow_queries.sql filters on back out of the data.

    Deriving them with SQL rather than from the generator's internal state means
    a literal can never be baked into slow_queries.sql that does not actually
    match a row -- a query that accidentally matches nothing would benchmark as
    trivially fast and quietly invalidate the whole seed set.
    """
    params: dict[str, str] = {
        "q1_sku": f"SKU-{PARAM_PRODUCT_Q1:07d}",
        "q2_country": RARE_COUNTRY,
    }
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM order_items WHERE sku = %s", (params["q1_sku"],))
        row = cur.fetchone()
        assert row is not None
        params["q1_matching_rows"] = str(row[0])

        cur.execute("SELECT count(*) FROM users WHERE country = %s", (RARE_COUNTRY,))
        row = cur.fetchone()
        assert row is not None
        params["q2_matching_users"] = str(row[0])

        # Query 3: a user with several orders, so lower(email_snapshot) matches
        # a handful of rows rather than exactly one.
        cur.execute(
            """
            SELECT o.email_snapshot, count(*)
            FROM orders o
            WHERE o.user_id BETWEEN %s AND %s
            GROUP BY o.email_snapshot
            HAVING count(*) >= 4
            ORDER BY count(*) DESC, o.email_snapshot
            LIMIT 1
            """,
            (PARAM_USER_Q3, PARAM_USER_Q3 + 2_000),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("no user in the sample window has >= 4 orders")
        params["q3_email_snapshot"] = row[0]
        params["q3_matching_rows"] = str(row[1])

        # Query 4: a different user (also with orders) plus a real tracking number.
        cur.execute(
            """
            SELECT u.email, count(*)
            FROM orders o
            JOIN users u ON u.id = o.user_id
            WHERE o.user_id BETWEEN %s AND %s
            GROUP BY u.email
            HAVING count(*) >= 3
            ORDER BY count(*) DESC, u.email
            LIMIT 1
            """,
            (PARAM_USER_Q4, PARAM_USER_Q4 + 2_000),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("no user in the sample window has >= 3 orders")
        params["q4_email"] = row[0]
        params["q4_email_matching_orders"] = str(row[1])

        cur.execute(
            """
            SELECT tracking_number
            FROM orders
            WHERE id >= %s AND tracking_number IS NOT NULL
            ORDER BY id
            LIMIT 1
            """,
            (PARAM_ORDER_Q4,),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("no order carries a tracking number")
        params["q4_tracking_number"] = row[0]

    missing = [key for key, value in params.items() if not value]
    if missing:
        raise RuntimeError(f"failed to capture query literals: {missing}")
    return params


def load(dsn: str, target: str, capture_params: bool) -> dict[str, str]:
    emails: list[str] = []
    wall_start = time.perf_counter()

    with psycopg.connect(dsn, autocommit=False) as conn:
        _log(target, "applying schema.sql")
        with conn.cursor() as cur:
            cur.execute(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()

        load_start = time.perf_counter()
        copy_table(conn, target, "users", gen_users(emails), N_USERS)
        copy_table(conn, target, "products", gen_products(), N_PRODUCTS)
        copy_table(conn, target, "orders", gen_orders(emails), N_ORDERS)
        copy_table(conn, target, "order_items", gen_order_items(), N_ORDER_ITEMS)
        copy_table(conn, target, "reviews", gen_reviews(), N_REVIEWS)
        load_elapsed = time.perf_counter() - load_start

        # ANALYZE is not optional. Without statistics the planner falls back to
        # hardcoded selectivity guesses, picks arbitrary plans, and every
        # before/after number the verifier produces later is noise. VACUUM is
        # bundled because COPY leaves the visibility map cleared, which would
        # otherwise block index-only scans on indexes the agent proposes.
        conn.commit()

    with psycopg.connect(dsn, autocommit=True) as conn:
        analyze_start = time.perf_counter()
        for table in TABLE_COLUMNS:
            started = time.perf_counter()
            with conn.cursor() as cur:
                cur.execute(f"VACUUM (ANALYZE) {table}")
            _log(target, f"VACUUM ANALYZE {table:<12} {time.perf_counter() - started:6.1f}s")
        analyze_elapsed = time.perf_counter() - analyze_start

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT relname,
                       to_char(reltuples, 'FM999,999,999') AS est_rows,
                       pg_size_pretty(pg_total_relation_size(oid)) AS total_size
                FROM pg_class
                WHERE relname = ANY(%s) AND relkind = 'r'
                ORDER BY pg_total_relation_size(oid) DESC
                """,
                (list(TABLE_COLUMNS),),
            )
            for relname, est_rows, total_size in cur.fetchall():
                # reltuples is the planner's estimate, not an exact count. The
                # exact count is asserted per table in copy_table().
                _log(target, f"  {relname:<12} ~{est_rows:>11} rows  {total_size:>9}")

        params = capture_query_params(conn) if capture_params else {}

    total = time.perf_counter() - wall_start
    _log(
        target,
        f"DONE  copy={load_elapsed:.1f}s  vacuum_analyze={analyze_elapsed:.1f}s  total={total:.1f}s",
    )

    if params:
        PARAMS_PATH.write_text(json.dumps(params, indent=2) + "\n", encoding="utf-8")
        _log(target, f"wrote {PARAMS_PATH.name}: {json.dumps(params)}")
    return params


def _worker(dsn: str, target: str, capture: bool) -> None:
    try:
        load(dsn, target, capture)
    except Exception as exc:  # re-raised in the parent via exit code
        print(f"[{target}] FAILED: {exc!r}", file=sys.stderr, flush=True)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed the OptiQuery databases.")
    parser.add_argument(
        "--target",
        choices=("primary", "shadow", "both"),
        default="both",
        help="which database(s) to load; 'both' loads them in parallel processes",
    )
    args = parser.parse_args(argv)

    primary_dsn = os.environ.get(
        "PRIMARY_DSN", "postgresql://optiquery:optiquery@localhost:55432/optiquery"
    )
    shadow_dsn = os.environ.get(
        "SHADOW_DSN", "postgresql://optiquery:optiquery@localhost:55433/optiquery"
    )

    print(
        f"scale={SCALE}  users={N_USERS:,}  products={N_PRODUCTS:,}  "
        f"orders={N_ORDERS:,}  order_items={N_ORDER_ITEMS:,}  reviews={N_REVIEWS:,}",
        flush=True,
    )

    started = time.perf_counter()
    if args.target == "primary":
        load(primary_dsn, "primary", capture_params=True)
    elif args.target == "shadow":
        load(shadow_dsn, "shadow", capture_params=False)
    else:
        # Two independent Postgres instances on 16 cores: run them concurrently
        # so 'both' costs roughly the same wall time as 'primary' alone.
        # Only the primary worker writes query_params.json, to avoid a race.
        procs = [
            multiprocessing.Process(target=_worker, args=(primary_dsn, "primary", True)),
            multiprocessing.Process(target=_worker, args=(shadow_dsn, "shadow", False)),
        ]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join()
        failed = [p for p in procs if p.exitcode != 0]
        if failed:
            print("seed failed; see errors above", file=sys.stderr)
            return 1

    print(f"\nseed complete in {time.perf_counter() - started:.1f}s wall", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
