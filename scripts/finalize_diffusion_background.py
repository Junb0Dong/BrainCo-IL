"""Durable tmux watcher: finish evaluation/publication even if the training shell predates those commands."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    reports = root / "outputs/diffusion"
    run = root / "checkpoints/diffusion/diffusion_pouring_set_1103_delta_obs1"
    print("Waiting for the train window to finish; this watcher survives SSH disconnects.", flush=True)
    while True:
        pane = subprocess.run(
            ["tmux", "display-message", "-p", "-t", "brainco-diffusion:train", "#{pane_dead}"],
            capture_output=True,
            text=True,
            check=False,
        )
        if pane.returncode or pane.stdout.strip() == "1":
            break
        time.sleep(15)
    with (reports / "finalize.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads((run / "status.json").read_text())
        if state.get("state") != "complete" or state.get("step") != 16000:
            raise RuntimeError(f"Training did not complete; retaining checkpoints and logs: {state}")
        checkpoint = (run / "best").resolve()
        report_path = reports / "final-deployment.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            if report.get("ok") and report.get("source") == str(checkpoint):
                print(f"Already verified and published: {report['destination']}", flush=True)
                return
        env = {
            **os.environ,
            "UV_PROJECT_ENVIRONMENT": ".venv-diffusion",
            "CUDA_VISIBLE_DEVICES": "4",
            "OMP_NUM_THREADS": "4",
            "PYTHONUNBUFFERED": "1",
        }
        evaluation = reports / "final-eval.json"
        if not evaluation.exists() or json.loads(evaluation.read_text()).get("checkpoint") != str(checkpoint):
            subprocess.run(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "python",
                    "scripts/eval_diffusion.py",
                    "--checkpoint",
                    str(checkpoint),
                    "--max-samples",
                    "256",
                    "--output-json",
                    str(evaluation),
                ],
                env=env,
                check=True,
            )
        subprocess.run(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "scripts/deploy_diffusion_checkpoint.py",
                "--checkpoint",
                str(checkpoint),
                "--report-json",
                str(report_path),
            ],
            env=env,
            check=True,
        )
        print("Training, evaluation, checksum verification and revi conformance are complete.", flush=True)


if __name__ == "__main__":
    main()
