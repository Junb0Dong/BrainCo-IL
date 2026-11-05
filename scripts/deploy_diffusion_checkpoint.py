"""Publish a verified DP inference-only artifact to revi, without starting robot execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from pathlib import PurePosixPath
import shlex
import subprocess
import uuid

from openpi.diffusion.checkpoint import verify_artifact

HOST = "revi@10.3.22.142"
RELEASE = "/home/revi/BrainCo/BrainCo-IL-diffusion-20261104"
DEPLOY = "/home/revi/BrainCo/revo_deploy/vla_deploy"
DESTINATION = "/home/revi/BrainCo/pi05_ckpt/junbo"
REGISTRY = "/home/revi/.brainco/revo_deploy/inference_backends.yaml"
DATASET = "/home/revi/.cache/huggingface/lerobot/001/revotron_mit_revo3_mit_3cam_pouring_set_junbo_1103"


def publish(checkpoint: Path, *, registry: str, socket: str | None, staging_checkpoint: str | None = None) -> dict:
    checkpoint = checkpoint.expanduser().resolve()
    cfg = verify_artifact(checkpoint)
    if not checkpoint.name.isdigit():
        raise ValueError("checkpoint must resolve to a completed numeric step directory")
    run_directory = PurePosixPath(DESTINATION) / cfg.name
    staging_path = (
        PurePosixPath(staging_checkpoint)
        if staging_checkpoint is not None
        else run_directory / f".upload-{uuid.uuid4().hex}" / checkpoint.name
    )
    if (
        staging_path.parent.parent != run_directory
        or not staging_path.parent.name.startswith(".upload-")
        or staging_path.name != checkpoint.name
    ):
        raise ValueError("staging checkpoint must be <destination>/<run>/.upload-<id>/<step>")
    ssh = ["ssh", "-o", "BatchMode=yes"]
    if socket is not None:
        ssh.extend(["-S", socket])
    ssh.append(HOST)

    def remote(command):
        return subprocess.check_output([*ssh, command], text=True)

    final = f"{DESTINATION}/{cfg.name}/{checkpoint.name}"
    staging = str(staging_path)
    report_path = f"{RELEASE}/validation/{cfg.name}-{checkpoint.name}.json"
    remote(shlex.join(["mkdir", "-p", staging, f"{RELEASE}/validation"]))
    # Explicit inference allowlist: optimizer, live training weights, data,
    # logs and arbitrary additional files never leave the training machine.
    subprocess.run(
        [
            "rsync",
            "-a",
            "--no-owner",
            "--no-group",
            "--partial",
            "--partial-dir=.rsync-partial",
            "--include=/params/***",
            "--include=/assets/***",
            "--include=/train_config.yaml",
            "--include=/policy_contract.json",
            "--include=/inference_manifest.json",
            "--exclude=*",
            "-e",
            shlex.join(ssh[:-1]),
            f"{checkpoint}/",
            f"{HOST}:{staging}/",
        ],
        check=True,
    )
    # The manifest is generated and checked locally before transfer. Remote
    # validation compares each file to those exact source SHA-256 values.
    manifest = json.loads((checkpoint / "inference_manifest.json").read_text())
    verification = (
        "import json; from pathlib import Path; "
        "from openpi.diffusion.checkpoint import verify_artifact; "
        f"p=Path({staging!r}); verify_artifact(p); "
        f"assert json.loads((p/'inference_manifest.json').read_text()) == {manifest!r}; "
        "print('Source and destination checksums match')"
    )
    remote(shlex.join([f"{RELEASE}/deployments/revi/python-brainco", "-c", verification]))
    command = [
        "env",
        f"PYTHONPATH={DEPLOY}",
        f"{RELEASE}/.venv/bin/python",
        "-m",
        "vla_deploy.backend_conformance",
        "--backend",
        "brainco_il",
        "--backend-config",
        registry,
        "--checkpoint",
        staging,
        "--robot-profile",
        f"{DEPLOY}/config/robot_profiles/revotron_revo3.yaml",
        "--through",
        "infer",
        "--dataset",
        DATASET,
        "--output-json",
        report_path,
    ]
    remote(shlex.join(command))
    conformance = json.loads(remote(shlex.join(["cat", report_path])))
    if not conformance.get("ok"):
        raise RuntimeError(f"remote backend conformance failed: {conformance}")
    publish_code = f"""from pathlib import Path
import json, os
from openpi.diffusion.checkpoint import verify_artifact
staging, final = Path({staging!r}), Path({final!r})
if final.exists():
    verify_artifact(final)
    if json.loads((final/'inference_manifest.json').read_text()) != {manifest!r}:
        raise RuntimeError('Refusing to replace a different completed checkpoint')
else:
    os.rename(staging, final)
print(final)
"""
    remote(shlex.join([f"{RELEASE}/deployments/revi/python-brainco", "-c", publish_code]))
    return {
        "ok": True,
        "source": str(checkpoint),
        "destination": final,
        "inference_bytes": sum((checkpoint / name).stat().st_size for name in manifest),
        "source_manifest": manifest,
        "conformance": conformance,
        "registry": registry,
        "remote_report": report_path,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--registry", default=REGISTRY)
    parser.add_argument("--ssh-socket", help="Reuse an authenticated SSH control socket; otherwise use SSH keys/config")
    parser.add_argument(
        "--staging-checkpoint", help="Reuse an interrupted remote .upload directory; verifies source hashes"
    )
    parser.add_argument("--report-json", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = publish(
            args.checkpoint, registry=args.registry, socket=args.ssh_socket, staging_checkpoint=args.staging_checkpoint
        )
    except Exception as exc:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps({"ok": False, "error": str(exc)}, indent=2) + "\n")
        raise
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("ok", "destination", "inference_bytes")}, indent=2))
