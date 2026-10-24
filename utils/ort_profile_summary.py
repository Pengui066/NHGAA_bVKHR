"""Summarize an ORT profiling trace (chrome-trace JSON from
EnableProfiling) for the S4 stage-0 inventory: top-10 operator costs and
the input/output copy share of the session.

Usage:
    python utils/ort_profile_summary.py <profile.json>
"""
import argparse
import collections
import json
import sys
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profile", help="chrome-trace JSON from ORT EnableProfiling")
    options = parser.parse_args()

    with open(options.profile, encoding="utf-8") as f:
        data = json.load(f)

    # ORT writes either {"traceEvents": [...]} or a bare event list.
    events = data if isinstance(data, list) else data.get("traceEvents", [])
    # Node events are the per-op records; their dur is in microseconds.
    node_events = [e for e in events if e.get("cat") == "Node"
                   and "dur" in e and "name" in e]

    total_us = sum(e["dur"] for e in node_events)
    per_op = collections.defaultdict(float)
    per_type = collections.defaultdict(float)
    for e in node_events:
        name = e["name"]
        per_op[name] += e["dur"]
        # Strip the trailing _<number> suffix to get the operator kind.
        kind = re.sub(r"_\d+$", "", name)
        per_type[kind] += e["dur"]

    print(f"events: {len(node_events)} | total op time: {total_us / 1e6:.2f} s")
    print(f"\n=== top-10 operators by total duration ===")
    for name, us in sorted(per_op.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {name:40s} {us / 1e6:8.2f} s  ({100 * us / total_us:5.1f}%)")

    print(f"\n=== by operator kind (top-10) ===")
    for kind, us in sorted(per_type.items(), key=lambda kv: -kv[1])[:10]:
        print(f"  {kind:40s} {us / 1e6:8.2f} s  ({100 * us / total_us:5.1f}%)")

    copy_us = sum(v for k, v in per_type.items()
                  if k.startswith("Memcpy") or "Copy" in k)
    print(f"\ncopy (Memcpy/transfer) total: {copy_us / 1e6:.2f} s "
          f"({100 * copy_us / total_us:.1f}% of op time)")


if __name__ == "__main__":
    main()
