"""Canonical row serialisation and result checksums. No database required.

The checksum is the only thing standing between "this rewrite is 1400x faster"
and "this rewrite is 1400x faster and returns the wrong rows". These tests are
about the ways a naive implementation says two different result sets are equal.
"""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from app.verifier import canonical_row, canonical_value, result_checksum


def digest(rows: list[tuple]) -> str:
    return result_checksum(rows).sorted_digest


class TestCanonicalValue:
    def test_null_is_a_dedicated_sentinel(self) -> None:
        """NULL must not be spellable by any non-NULL value."""
        assert canonical_value(None) == "N:"
        assert canonical_value(None) != canonical_value("")
        assert canonical_value(None) != canonical_value("N:")
        assert canonical_value(None) != canonical_value("NULL")

    def test_bool_is_distinct_from_int(self) -> None:
        """bool is a subclass of int in Python; True must not serialise as 1."""
        assert canonical_value(True) != canonical_value(1)
        assert canonical_value(False) != canonical_value(0)

    def test_numeric_scale_is_normalised(self) -> None:
        """10.00 and 10.0 are the same number carrying different scale.

        A rewrite can legitimately change scale -- summing in a different order,
        or numeric widened through a join. Rejecting that would be a false
        failure.
        """
        assert canonical_value(Decimal("10.00")) == canonical_value(Decimal("10.0"))
        assert canonical_value(Decimal("10")) == canonical_value(Decimal("10.000"))
        assert canonical_value(Decimal("1E+2")) == canonical_value(Decimal("100"))

    def test_numeric_values_that_differ_still_differ(self) -> None:
        assert canonical_value(Decimal("10.01")) != canonical_value(Decimal("10.1"))

    def test_int_and_numeric_are_distinguishable(self) -> None:
        assert canonical_value(10) != canonical_value(Decimal("10"))

    def test_datetime_is_iso_formatted(self) -> None:
        stamp = datetime(2024, 3, 15, 12, 30, tzinfo=timezone.utc)
        assert canonical_value(stamp) == "T:2024-03-15T12:30:00+00:00"

    def test_dict_key_order_does_not_matter(self) -> None:
        """jsonb does not preserve key order; two equal values must hash equal."""
        assert canonical_value({"a": 1, "b": 2}) == canonical_value({"b": 2, "a": 1})

    def test_nested_containers_round_trip(self) -> None:
        assert canonical_value([1, [2, None]]) != canonical_value([1, [2, 0]])


class TestFraming:
    def test_field_boundaries_are_unambiguous(self) -> None:
        """('a','b') and ('ab',) must not serialise identically.

        This is the failure any separator-based encoding has, because the
        separator can occur inside the data.
        """
        assert canonical_row(("a", "b")) != canonical_row(("ab",))

    def test_a_separator_inside_data_cannot_forge_a_boundary(self) -> None:
        assert canonical_row(("a|b",)) != canonical_row(("a", "b"))
        assert canonical_row(("1|S:x",)) != canonical_row(("x",))

    def test_null_shifting_across_columns_is_detected(self) -> None:
        assert canonical_row(("a", None)) != canonical_row((None, "a"))

    def test_row_boundaries_are_unambiguous(self) -> None:
        assert digest([("a",), ("b",)]) != digest([("ab",)])


class TestResultChecksum:
    def test_row_order_does_not_affect_the_equivalence_digest(self) -> None:
        """A different plan may emit the same rows in a different order."""
        assert digest([(1, "a"), (2, "b")]) == digest([(2, "b"), (1, "a")])

    def test_row_order_does_affect_the_order_sensitive_digest(self) -> None:
        """Reported alongside, never used to reject. See Checksum's docstring."""
        first = result_checksum([(1, "a"), (2, "b")])
        second = result_checksum([(2, "b"), (1, "a")])
        assert first.sorted_digest == second.sorted_digest
        assert first.ordered_digest != second.ordered_digest

    def test_same_row_count_different_rows_is_caught(self) -> None:
        """The reason row counts are not equivalence."""
        left = [(i, "x") for i in range(15)]
        right = [(i + 100, "x") for i in range(15)]
        assert len(left) == len(right)
        assert digest(left) != digest(right)

    def test_duplicate_rows_change_the_digest(self) -> None:
        """A UNION ALL rewrite missing its anti-predicate duplicates rows."""
        assert digest([(1,), (2,)]) != digest([(1,), (1,), (2,)])

    def test_empty_result_has_a_stable_digest(self) -> None:
        """Zero rows is a legitimate result and must compare equal to itself."""
        assert digest([]) == digest([])
        assert digest([]) != digest([(None,)])

    def test_null_versus_empty_string(self) -> None:
        assert digest([(None,)]) != digest([("",)])

    def test_identical_input_is_deterministic_across_calls(self) -> None:
        rows = [(1, Decimal("2.50"), None, datetime(2024, 1, 1, tzinfo=timezone.utc))]
        assert digest(rows) == digest(rows)

    def test_digest_is_sha256_hex(self) -> None:
        value = digest([(1,)])
        assert len(value) == 64
        assert set(value) <= set("0123456789abcdef")

    def test_row_count_is_reported(self) -> None:
        assert result_checksum([(1,), (2,), (3,)]).row_count == 3


class TestScaleNormalisationDoesNotOverReach:
    @pytest.mark.parametrize(
        "left,right",
        [
            (Decimal("0.1"), Decimal("0.10000000000000001")),
            (Decimal("-1.5"), Decimal("1.5")),
            (Decimal("0"), Decimal("0.0000001")),
        ],
    )
    def test_genuinely_different_numerics_are_not_collapsed(
        self, left: Decimal, right: Decimal
    ) -> None:
        assert canonical_value(left) != canonical_value(right)
