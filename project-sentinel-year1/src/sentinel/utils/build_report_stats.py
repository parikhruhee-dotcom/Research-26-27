"""`make report` target: runs the full pytest suite one final time and writes
results/TEST_SUMMARY.json (pass/fail counts) so the scientific report and
reproducibility artifact can state honest, up-to-date numbers.

Run: python -m sentinel.utils.build_report_stats
"""
from __future__ import annotations

import json
import subprocess

from sentinel.utils.config import repo_path
from sentinel.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> dict:
    result = subprocess.run(
        ["python3", "-m", "pytest", "tests/", "-v", "--tb=short"],
        cwd=str(repo_path()), capture_output=True, text=True, timeout=600,
    )
    output = result.stdout + result.stderr
    summary_line = [l for l in output.splitlines() if " passed" in l or " failed" in l or " error" in l]
    tail = summary_line[-1] if summary_line else "no summary line found"

    n_passed = output.count(" PASSED")
    n_failed = output.count(" FAILED")
    n_skipped = output.count(" SKIPPED")
    n_error = output.count(" ERROR")

    summary = {
        "return_code": result.returncode, "summary_line": tail,
        "n_passed": n_passed, "n_failed": n_failed, "n_skipped": n_skipped, "n_error": n_error,
        "all_passed": result.returncode == 0,
    }
    with open(repo_path("results", "TEST_SUMMARY.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(repo_path("results", "pytest_full_output.log"), "w") as fh:
        fh.write(output)

    logger.info(f"pytest: {tail}")
    return summary


if __name__ == "__main__":
    main()
