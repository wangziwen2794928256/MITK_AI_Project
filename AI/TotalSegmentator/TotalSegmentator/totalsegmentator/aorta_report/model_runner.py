from __future__ import annotations

import asyncio
import json
import subprocess
import time
from pathlib import Path


TOTAL_ROIS = [
    "aorta",
    "brachiocephalic_trunk",
    "subclavian_artery_left",
    "common_carotid_artery_left",
    "iliac_artery_left",
    "iliac_artery_right",
    "vertebrae_T12",
]

MODEL_RUNS = [
    (
        "Running TotalSegmentator - aorta",
        "  Skipping TotalSeg (already exists)",
        ["brachiocephalic_trunk.nii.gz"],
        ["-ta", "total", "-rs", *TOTAL_ROIS],
    ),
    (
        "Running T716 TF lumen",
        "  Skipping T716 (already exists)",
        ["aorta_true_lumen.nii.gz", "aorta_false_lumen.nii.gz"],
        ["-ta", "aortic_dissection"],
    ),
    (
        "Running T713 annulus proper",
        "  Skipping T713 (already exists)",
        ["annulus_proper.nii.gz", "sinotubular_junction.nii.gz"],
        ["-ta", "aorta_annulus"],
    ),
    (
        "Running T710 renal arteries",
        "  Skipping T710 (already exists)",
        ["celiac_trunk.nii.gz", "superior_mesenteric_artery.nii.gz", "renal_arteries.nii.gz"],
        ["-ta", "renal_arteries"],
    ),
]


def _command(ct_path: Path, output_dir: Path, task_args: list[str]) -> list[str]:
    return [
        "TotalSegmentator",
        "-i",
        str(ct_path),
        "-o",
        str(output_dir),
        "-ns",
        "1",
        *task_args,
    ]


def _outputs_exist(output_dir: Path, names: list[str]) -> bool:
    return all((output_dir / name).exists() for name in names)


def _check_outputs(output_dir: Path, names: list[str], description: str) -> None:
    missing = [str(output_dir / name) for name in names if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"{description} did not create the expected output files:\n  - "
            + "\n  - ".join(missing)
        )


def _run(command: list[str], description: str) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode:
        raise RuntimeError(f"{description} failed with exit code {completed.returncode}")


def get_aorta_fast(ct_path, tmp_dir, logger, totalseg_version="v2", host="local"):
    """Create the fast aorta/heart masks used for cropping."""
    if (tmp_dir / "aorta.nii.gz").exists():
        logger.info("  Skipping TotalSeg aorta fast (already exists)")
        return
    if host != "local":
        raise ValueError("The migrated aorta report supports local model execution only.")

    heart_name = "heart_myocardium" if totalseg_version == "v1" else "heart"
    command = _command(
        Path(ct_path),
        Path(tmp_dir),
        ["-ta", "total", "-rs", "aorta", heart_name, "--fast"],
    )
    _run(command, "TotalSegmentator aorta fast")
    _check_outputs(Path(tmp_dir), ["aorta.nii.gz", f"{heart_name}.nii.gz"], "TotalSegmentator aorta fast")


def run_models_consecutive(ct_path, tmp_dir, logger):
    """Run the four registered TotalSegmentator tasks sequentially."""
    started = time.time()
    ct_path, tmp_dir = Path(ct_path), Path(tmp_dir)
    for status, skip_message, outputs, task_args in MODEL_RUNS:
        yield status
        if _outputs_exist(tmp_dir, outputs):
            logger.info(skip_message)
            continue
        _run(_command(ct_path, tmp_dir, task_args), status)
        _check_outputs(tmp_dir, outputs, status)
    print(f"Models done (took: {time.time() - started:.2f}s)")
    yield "Models done"


async def _run_async(command: list[str], description: str) -> None:
    process = await asyncio.create_subprocess_exec(*command)
    return_code = await process.wait()
    if return_code:
        raise RuntimeError(f"{description} failed with exit code {return_code}")


async def run_models_parallel(ct_path, tmp_dir, logger, host="local"):
    """Run independent registered tasks concurrently, writing per-class masks."""
    if host != "local":
        raise ValueError("The migrated aorta report supports local model execution only.")

    print("Running Models - ASYNC")
    started = time.time()
    ct_path, tmp_dir = Path(ct_path), Path(tmp_dir)
    pending = []
    pending_metadata = []
    for status, skip_message, outputs, task_args in MODEL_RUNS:
        if _outputs_exist(tmp_dir, outputs):
            logger.info(skip_message)
            continue
        pending.append(_run_async(_command(ct_path, tmp_dir, task_args), status))
        pending_metadata.append((status, outputs))
    if pending:
        await asyncio.gather(*pending)
    for status, outputs in pending_metadata:
        _check_outputs(tmp_dir, outputs, status)
    print(f"Models ASYNC done (took: {time.time() - started:.2f}s)")


def get_contrast_phase(ct_path, tmp_dir, logger, host="local"):
    """Run TotalSegmentator's contrast-phase classifier with source-compatible caching."""
    if host != "local":
        raise ValueError("The migrated aorta report supports local model execution only.")
    started = time.time()
    output_path = Path(tmp_dir) / "contrast_phase.json"
    if output_path.exists():
        logger.info("  Skipping totalseg_get_phase (already exists)")
    else:
        completed = subprocess.run(
            ["totalseg_get_phase", "-i", str(ct_path), "-o", str(output_path), "-q"],
            check=False,
        )
        if completed.returncode:
            raise RuntimeError(
                f"totalseg_get_phase failed with exit code {completed.returncode}"
            )
    with output_path.open() as file:
        result = json.load(file)
    print(f"Contrast phase done (took: {time.time() - started:.2f}s)")
    return result["phase"]
