import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REFERENCE_DIR = Path(__file__).parent / "reference_files" / "aorta_report"
EXAMPLE_CT = REFERENCE_DIR / "example_ct.nii.gz"
EXAMPLE_NODEINFO = REFERENCE_DIR / "example_nodeinfo.json"
REFERENCE_JSON = REFERENCE_DIR / "example_ct_report" / "aorta_report.json"
CACHED_MODELS_DIR = REFERENCE_DIR / "test_result" / "models"


def dicts_almost_equal(d1, d2):
    """
    Check if two nested dicts or lists are equal.

    For integers, allow a difference of 1.
    For floats, allow a difference of 2.
    """
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            print(f"keys not equal: {d1.keys()}, {d2.keys()}")
            return False

        for key in d1:
            if not dicts_almost_equal(d1[key], d2[key]):
                print(f"mismatch for key: {key}")
                return False

    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            print(f"list lengths not equal: {len(d1)}, {len(d2)}")
            return False

        for i in range(len(d1)):
            if not dicts_almost_equal(d1[i], d2[i]):
                print(f"list item mismatch at index {i}")
                return False

    elif isinstance(d1, int) and isinstance(d2, int):
        if abs(d1 - d2) > 1:
            print(f"integer mismatch: d1: {d1}, d2: {d2} -> diff more than 1")
            return False

    elif isinstance(d1, float) and isinstance(d2, float):
        if abs(d1 - d2) > 2:
            print(f"float mismatch: d1: {d1}, d2: {d2} -> diff more than 2")
            return False

    elif d1 != d2:
        print(f"mismatch: d1: {d1}, d2: {d2} -> not equal")
        return False

    return True


def remove_keys_with_prefix(d, prefixes):
    """Return a copy without keys starting with any of the prefixes."""
    if isinstance(d, dict):
        return {
            key: remove_keys_with_prefix(value, prefixes)
            for key, value in d.items()
            if not any(key.startswith(prefix) for prefix in prefixes)
        }
    if isinstance(d, list):
        return [remove_keys_with_prefix(item, prefixes) for item in d]
    return d


@pytest.fixture(scope="module")
def aorta_report_result(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("aorta_report")
    output_nifti = tmp_path / "aorta_report.nii.gz"
    output_json = tmp_path / "aorta_report.json"
    output_log = tmp_path / "aorta_report.log"

    shutil.copytree(CACHED_MODELS_DIR, tmp_path / "models")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "totalsegmentator.bin.totalseg_aorta_report",
            "-i",
            str(EXAMPLE_CT),
            "-n",
            str(EXAMPLE_NODEINFO),
            "-o",
            str(output_nifti),
            "-j",
            str(output_json),
            "-l",
            str(output_log),
            "-tmp",
            str(tmp_path),
            "-a",
            "aorta.nii.gz",
            "-an",
            "annulus_proper.nii.gz",
            "-t",
            "basic",
            "--run_models",
            "--erosion",
            "--debug",
        ],
        check=True,
    )

    return {
        "output_nifti": output_nifti,
        "output_json": output_json,
        "output_log": output_log,
    }


def test_aorta_report_logs(aorta_report_result):
    logs = aorta_report_result["output_log"].read_text()
    required_contents = [
        "already cropped (skipping)",
        "Skipping TotalSeg (already exists)",
        "Skipping T716 (already exists)",
        "Skipping T713 (already exists)",
        "Skipping T710 (already exists)",
        "Contrast phase: arterial_early",
        "WARNING: iliac is empty!",
        "WARNING: iliac is (almost) empty or has no intersection with aorta! Setting to empty.",
        "WARNING: landmark 11 is empty because iliac is empty!",
        "Processing landmark 1",
        "WARNING: max_dia of aorta section aorta_descending is larger than max_dia from landmarks (4",
        "WARNING: aorta_abdominal is empty!",
    ]

    for content in required_contents:
        assert content in logs, f"required content not found in logs: {content}"


def test_aorta_report_json(aorta_report_result):
    with open(REFERENCE_JSON) as f:
        data_ref = json.load(f)
    with open(aorta_report_result["output_json"]) as f:
        data_new = json.load(f)

    data_ref_filtered = remove_keys_with_prefix(data_ref, ["start", "end"])
    data_new_filtered = remove_keys_with_prefix(data_new, ["start", "end"])

    assert dicts_almost_equal(data_ref_filtered, data_new_filtered)


def test_aorta_report_files_exist(aorta_report_result):
    assert aorta_report_result["output_nifti"].is_file()
