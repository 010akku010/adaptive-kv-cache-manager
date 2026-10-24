"""
demo_page.py

Builds the self-contained browser demo (docs/demo/index.html) by recording a few
scenarios with ``src.demo`` and embedding them in docs/demo/template.html.

Usage:
    python -m src.demo_page            # writes docs/demo/index.html
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from src.demo import DemoSetup, record_pair

SCENARIOS: List[Dict[str, Any]] = [
    {
        "label": "Uniform traffic",
        "note": "Every conversation is equally likely to return, so recency alone tells a policy little. Plenty of host memory.",
        "setup": DemoSetup(workload="uniform", seed=0),
    },
    {
        "label": "Long documents",
        "note": "A few very long conversations (document or RAG style): large block sets and high GPU pressure.",
        "setup": DemoSetup(workload="long_context", seed=0),
    },
    {
        "label": "Scarce host memory",
        "note": "Popular conversations return often, but the CPU tier holds only 512 MB, so offloading is no longer free.",
        "setup": DemoSetup(workload="zipf", seed=0, cpu_pool_mb=512.0),
    },
]

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "demo")


def build_data() -> Dict[str, Any]:
    scenarios = []
    for sc in SCENARIOS:
        pair = record_pair(sc["setup"])
        scenarios.append({"label": sc["label"], "note": sc["note"], "runs": pair["runs"]})
    return {"scenarios": scenarios}


STANDALONE_HEAD = (
    '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width, initial-scale=1">'
    "<style>:root{color-scheme:light}body{margin:0}</style></head><body>\n"
)
STANDALONE_TAIL = "\n</body></html>\n"


def render_page(data: Dict[str, Any], template_path: Optional[str] = None, standalone: bool = True) -> str:
    """Embed ``data`` in the template. ``standalone`` adds the html/head/body wrapper for opening as a file."""
    with open(template_path or os.path.join(HERE, "template.html")) as fh:
        template = fh.read()
    if "/*DATA*/null" not in template:
        raise ValueError("template has no /*DATA*/null placeholder")
    body = template.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")), 1)
    return STANDALONE_HEAD + body + STANDALONE_TAIL if standalone else body


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    out = os.path.join(HERE, "index.html")
    data = build_data()
    page = render_page(data)
    with open(out, "w") as fh:
        fh.write(page)
    print(f"wrote {out} ({len(page) / 1024:.0f} KB)")
    if args:  # optional: also write the unwrapped body (for publishing inside a host page)
        with open(args[0], "w") as fh:
            fh.write(render_page(data, standalone=False))


if __name__ == "__main__":
    main()
