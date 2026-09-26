"""CLI: polarity-pool Stage B."""

from __future__ import annotations

import sys

from steering.xy_control.stagebc_polarity import main


if __name__ == "__main__":
    sys.exit(main("b"))
