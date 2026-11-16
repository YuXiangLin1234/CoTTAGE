import argparse
import json
import os
import re
from collections import defaultdict

from scipy.stats import pearsonr, spearmanr

SCORE_RE = re.compile(r"<score>\s*([0-9]+(?:\.[0-9]+)?)\s*</score>")
SHARD_RE = re.compile(r"\.shard\d+(?=\.json$)")


def _label(path):
    return os.path.join(*SHARD_RE.sub("", path).split(os.sep)[-2:])


def collect(paths):
    groups = defaultdict(lambda: {"pairs": [], "unparsed": 0})
    for path in paths:
        group = groups[_label(path)]
        for row in json.load(open(path, encoding="utf-8")):
            match = SCORE_RE.search(row.get("response") or "")
            if match is None:
                group["unparsed"] += 1
            elif row.get("gt") is not None:
                group["pairs"].append((float(row["gt"]), float(match.group(1))))
    return groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="result JSON files written by run_inference.py")
    args = parser.parse_args()

    print(f"{'run':<48}{'n':>7}{'LCC':>8}{'SRCC':>8}{'unparsed':>10}")
    for name, group in sorted(collect(args.results).items()):
        pairs = group["pairs"]
        if len(pairs) < 2:
            print(f"{name:<48}{len(pairs):>7}{'-':>8}{'-':>8}{group['unparsed']:>10}")
            continue
        gt, pred = zip(*pairs)
        print(f"{name:<48}{len(pairs):>7}{pearsonr(gt, pred)[0]:>8.3f}"
              f"{spearmanr(gt, pred)[0]:>8.3f}{group['unparsed']:>10}")


if __name__ == "__main__":
    main()
