"""Per-SOC XY-control Stage B: capability selection on MMLU-Pro domain_val."""

from __future__ import annotations

import sys

from steering.xy_control.stagebc import main


if __name__ == "__main__":
    sys.exit(main("b"))
