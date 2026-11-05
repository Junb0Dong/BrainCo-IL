"""Compare recorded-observation outputs before replacing an inference backend.

Run in the revo_deploy Python environment, with its package on PYTHONPATH.
No ROS nodes or robot commands are started.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from vla_deploy.common.inferencer_manager import PolicyInferencerManager
from vla_deploy.common.named_io import DatasetNamedObservationSource


def infer(registry, checkpoint, dataset, *, reverse_names, strategy):
    manager = PolicyInferencerManager(registry, "brainco_il", checkpoint, source_type="dataset")
    try:
        capabilities = manager.capabilities
        source = DatasetNamedObservationSource(dataset, manager.contract, start=0, end=1, loop=False)
        observation = source.get_observation()
        if reverse_names:
            for group in observation["joint_groups"].values():
                group["names"] = list(reversed(group["names"]))
                group["positions"] = np.asarray(group["positions"])[::-1].copy()
        manager.load(inference_config={"strategy": strategy})
        result = manager.infer(1, observation)["actions"]
        if strategy == "rtc":
            previous = np.concatenate([result[group] for group in manager.contract.raw["output_joint_groups"]], axis=1)
            result = manager.infer(
                2,
                observation,
                rtc_context={
                    "prev_chunk_left_over": previous[len(previous) // 2 :],
                    "predicted_delay_steps": 1,
                },
            )["actions"]
        return result, capabilities
    finally:
        manager.close()


def compare(old, new, checkpoint, dataset):
    reference, caps = infer(old, checkpoint, dataset, reverse_names=False, strategy="standard")
    actual, new_caps = infer(new, checkpoint, dataset, reverse_names=True, strategy="standard")
    if caps != new_caps:
        raise AssertionError(f"capabilities changed for {checkpoint}")

    def differences(left, right):
        if left.keys() != right.keys():
            raise AssertionError("action groups changed")
        errors = {}
        for group in left:
            np.testing.assert_allclose(left[group], right[group], rtol=0, atol=1e-5)
            if not np.isfinite(right[group]).all():
                raise AssertionError("non-finite actions")
            errors[group] = float(np.max(np.abs(left[group] - right[group])))
        return errors

    result = {
        "checkpoint": checkpoint,
        "standard_max_abs_difference": differences(reference, actual),
        "capabilities": caps,
        "candidate_input_names": "reversed",
    }
    if "rtc" in caps.get("inference_strategies", []):
        reference, _ = infer(old, checkpoint, dataset, reverse_names=False, strategy="rtc")
        actual, _ = infer(new, checkpoint, dataset, reverse_names=True, strategy="rtc")
        result["rtc_max_abs_difference"] = differences(reference, actual)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-registry", required=True)
    parser.add_argument("--new-registry", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    report = {"ok": False, "comparisons": []}
    try:
        for checkpoint in args.checkpoint:
            print(f"Comparing {checkpoint}", flush=True)
            report["comparisons"].append(compare(args.old_registry, args.new_registry, checkpoint, args.dataset))
        report["ok"] = True
    finally:
        Path(args.output_json).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
