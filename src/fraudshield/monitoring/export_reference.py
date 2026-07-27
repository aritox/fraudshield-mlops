"""Export the frozen aggregate training reference for Phase 2D.2."""

from __future__ import annotations

import argparse
import json

from fraudshield.monitoring.config import load_monitoring_config
from fraudshield.monitoring.reference import export_reference


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    result = export_reference(load_monitoring_config())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
