"""Per-SOC XY-control Stage C: frozen confirmation on test + domain_test."""

from __future__ import annotations

import sys

from steering.xy_control.stagebc import main


if __name__ == "__main__":
    sys.exit(main("c"))
