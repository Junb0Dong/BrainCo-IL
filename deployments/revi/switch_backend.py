"""Switch only the registered BrainCo wrapper, preserving other registry content.

Run with the candidate release's Python on revi after staging conformance passes.
Use --rollback <switch-report.json> to restore the preceding wrapper.
"""

import argparse
import copy
import datetime as dt
import json
import os
from pathlib import Path
import tempfile

import yaml

REGISTRY = Path("/home/revi/.brainco/revo_deploy/inference_backends.yaml")
OLD_WRAPPER = (
    "/home/revi/BrainCo/BrainCo-IL-junling-pi05-20261028/deployments/revi-junling-epoch3-20261028/python-epoch3"
)
NEW_WRAPPER = str(Path(__file__).resolve().with_name("python-brainco"))


def switch(registry: Path, expected: str, replacement: str, report_path: Path) -> None:
    original = registry.read_text()
    before = yaml.safe_load(original)
    command = before["backends"]["brainco_il"]["command"]
    if command != [expected, "-m", "openpi.deploy.policy_inferencer"]:
        raise RuntimeError(f"Registry command changed; refusing to overwrite: {command!r}")
    if not os.access(replacement, os.X_OK):
        raise FileNotFoundError(f"Replacement wrapper is not executable: {replacement}")

    # Replace only this scalar so comments, anchors and other backends stay intact.
    node = yaml.compose(original)
    for key in ("backends", "brainco_il", "command"):
        node = next(value for name, value in node.value if name.value == key)
    scalar = node.value[0]
    updated = original[: scalar.start_mark.index] + json.dumps(replacement) + original[scalar.end_mark.index :]
    expected_config = copy.deepcopy(before)
    expected_config["backends"]["brainco_il"]["command"][0] = replacement
    if yaml.safe_load(updated) != expected_config:
        raise RuntimeError("Unexpected changes to registry")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    backup = report_path.with_suffix(".before.yaml")
    with backup.open("x") as stream:
        stream.write(original)
    report = {
        "state": "prepared",
        "registry": str(registry),
        "backup": str(backup),
        "before_wrapper": expected,
        "after_wrapper": replacement,
        "time_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    with report_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    fd, temporary = tempfile.mkstemp(prefix=".backend-switch-", dir=registry.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(updated)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, registry.stat().st_mode & 0o777)
        if registry.read_text() != original:
            raise RuntimeError("Registry changed during preparation; retry against current content")
        os.replace(temporary, registry)
    finally:
        Path(temporary).unlink(missing_ok=True)
    report["state"] = "switched"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--rollback", type=Path)
    args = parser.parse_args()
    if args.rollback:
        previous = json.loads(args.rollback.read_text())
        if previous["registry"] != str(args.registry):
            raise ValueError("Rollback report belongs to a different registry")
        switch(args.registry, previous["after_wrapper"], previous["before_wrapper"], args.report)
    else:
        switch(args.registry, OLD_WRAPPER, NEW_WRAPPER, args.report)
