"""Print the result of every warehouse query, so they can be eyeballed by hand."""

from __future__ import annotations

import pandas as pd

from selector.warehouse import queries

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 120)


def main() -> None:
    calls = [
        ("top_artists", lambda: queries.top_artists(limit=10)),
        ("binged_then_abandoned", lambda: queries.binged_then_abandoned().head(10)),
        ("skip_offenders", lambda: queries.skip_offenders().head(10)),
        ("listening_clock", lambda: queries.listening_clock().head(10)),
        ("taste_drift", lambda: queries.taste_drift()),
        ("rediscovery_candidates", lambda: queries.rediscovery_candidates().head(10)),
    ]

    for name, call in calls:
        print(f"\n=== {name} ===")
        print(call())


if __name__ == "__main__":
    main()
