"""Check a round's query CSV against the portal format before submission.

Each query: x1-x2-...-xd, every value written as 0.dddddd (six decimals), in [0, 1).
Usage:  python src/validate_query.py rounds/week10/Week10_queries.csv
"""
import csv, re, sys
DIMS = {1: 2, 2: 2, 3: 3, 4: 4, 5: 4, 6: 5, 7: 6, 8: 8}
PAT = re.compile(r"^0\.\d{6}$")

def check(path):
    rows = list(csv.DictReader(open(path)))
    assert [int(r["function"]) for r in rows] == list(range(1, 9)), "need functions 1..8 in order"
    for r in rows:
        parts = r["query_string"].split("-")
        i = int(r["function"])
        assert len(parts) == DIMS[i], f"F{i}: {len(parts)} values, expected {DIMS[i]}"
        bad = [p for p in parts if not PAT.match(p)]
        assert not bad, f"F{i}: badly formatted {bad}"
    print(f"{path}: OK (8 queries)")

if __name__ == "__main__":
    for p in sys.argv[1:]:
        check(p)
