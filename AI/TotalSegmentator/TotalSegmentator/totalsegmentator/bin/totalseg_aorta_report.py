import argparse
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from totalsegmentator.aorta_report import VERSION
from totalsegmentator.aorta_report.report import create_aorta_report
from totalsegmentator.reporting import save_runtime, setup_logger
from totalsegmentator.serialization_utils import (
    convert_to_serializable,
    decompress_and_deserialize,
)


def _load_metadata(path, logger):
    if path is not None and path.exists():
        with path.open() as file:
            metadata = json.load(file, strict=False)
        for source, target in (
            ("PatientsAge", "PatientAge"),
            ("PatientsSex", "PatientSex"),
            ("PatientsBirthDate", "PatientBirthDate"),
        ):
            if source in metadata:
                metadata[target] = metadata.pop(source)
    else:
        logger.info(
            "WARNING: Could not find nodeinfo.json. Using default metadata."
        )
        metadata = {
            "PatientName": "Unknown",
            "PatientSex": "M",
            "PatientAge": "083Y",
            "PatientBirthDate": "19000101",
            "PatientID": "Unknown",
            "StudyID": "Unknown",
            "StudyDescription": "Unknown",
            "StudyDate": "19000101",
        }
    metadata.setdefault("PatientAge", "099Y")
    if "W" in str(metadata["PatientAge"]):
        logger.info(
            "WARNING: Patient age is in weeks. Patient too young. Stopping."
        )
        return None
    metadata["PatientAge"] = int(str(metadata["PatientAge"]).replace("Y", ""))
    metadata["version"] = VERSION
    return metadata


def _build_parser():
    parser = argparse.ArgumentParser(description="Generate aorta report.")

    def resolved_path(value):
        return Path(value).resolve()

    parser.add_argument("-i", "--ct_path", type=resolved_path, required=True,
                        help="Path to ct file.")
    parser.add_argument("-rt", "--rois_totalseg", type=resolved_path, default=None,
                        help="Path to directory containing the totalseg rois.")
    parser.add_argument("-rd", "--rois_details", type=resolved_path, default=None,
                        help="Path to directory containing the detailed aorta analysis specific rois.")
    parser.add_argument("-n", "--nodeinfo", type=resolved_path,
                        help="Optional path to nodeinfo file.")
    parser.add_argument("-o", "--output_nifti", type=resolved_path, required=True,
                        help="Path to output nifti file")
    parser.add_argument("-j", "--output_json", type=resolved_path, required=True,
                        help="Path to output json file")
    parser.add_argument("-l", "--output_log", type=resolved_path, required=True,
                        help="Path to output log file")
    parser.add_argument("-tmp", "--tmp_dir", type=resolved_path,
                        help="Path to tmp dir. If not set, then use system tmp dir.")
    parser.add_argument("-c", "--cpr", type=resolved_path, default=None,
                        help="Path to CPR output png file.")
    parser.add_argument("-ca", "--cpr_animated", type=resolved_path, default=None,
                        help="Path to animated CPR output nifti file (.nii.gz).")
    parser.add_argument("-a", "--aorta_fn", default="aorta.nii.gz",
                        help="Filename of the aorta mask.")
    parser.add_argument("-an", "--annulus_fn", default="annulus_proper.nii.gz",
                        help="Filename of the annulus mask.")
    parser.add_argument("-rm", "--run_models", action="store_true",
                        help="Run all models from within this code instead of in extra code.")
    parser.add_argument("-mp", "--models_parallel", action="store_true",
                        help="Run all models in parallel. Faster, but needs more RAM + GPU memory.")
    parser.add_argument("-t", "--test", default="None", help="Define which test to run.")
    parser.add_argument("-e", "--erosion", action="store_true",
                        help="Erode aorta by 0.8mm to make aorta fit better.")
    parser.add_argument("-d", "--debug", action="store_true",
                        help="If debug use other tmp dir and same more intermediate outputs.")
    parser.add_argument("-sd", "--skip_dissection", action="store_true",
                        help="Skip using true/false lumen segmentation even if available.")
    parser.add_argument("-r", "--save_runtime", action="store_true",
                        help="Save runtime to META/runtime.json file.")
    parser.add_argument("--version", action="version", version=VERSION)
    return parser

"""
Run for one test case:

cd /mnt/nvme/data/test_data/aorta_report/28500925
totalseg_aorta_report \
  -i ct_15mm.nii.gz \
  -rt roi \
  -rd roi_cropped_aorta \
  -o aorta_report/aorta_report.nii.gz \
  -j aorta_report/aorta_report.json \
  -l aorta_report/aorta_report.txt \
  -tmp tmp_aorta_report \
  -c aorta_report/cpr.png \
  -a organ_aorta_T264.nii.gz \
  -an annulus.nii.gz \
  -d
"""
def main():
    args = _build_parser().parse_args()
    if args.debug:
        print("Running in DEBUG mode.")
        tmp_dir = args.tmp_dir.absolute()
        delete_tmp = delete_aux_files = False
    elif args.tmp_dir:
        tmp_dir = args.tmp_dir.absolute()
        delete_tmp, delete_aux_files = False, True
    else:
        tmp_dir = Path(tempfile.mkdtemp())
        delete_tmp = delete_aux_files = True
    tmp_dir.mkdir(exist_ok=True)
    args.output_nifti.parent.mkdir(exist_ok=True)
    logger = setup_logger(
        args.output_log, name=f"totalseg_aorta_report.{args.output_log}"
    )
    started = time.time()
    logger.info("Getting metadata...")
    metadata = _load_metadata(args.nodeinfo, logger)
    if metadata is None:
        return

    ct_img = nib.load(args.ct_path)
    f_type = "nii" if args.ct_path.suffix == ".nii" else "niigz"
    linear_affine = ct_img.affine[:3, :3].copy()
    if linear_affine.min() < 0:
        logger.info("WARNING: Affine contains negative values!")
    np.fill_diagonal(linear_affine, 0)
    if linear_affine.max() > 0.01:
        logger.info("WARNING: Affine contains significant rotation!")
    shape = ct_img.shape
    if metadata["PatientAge"] < 5 or metadata["PatientAge"] > 150:
        logger.info("WARNING: Patient age is not in range [5, 150]. Stopping.")
        return
    if max(shape) > 1500:
        logger.info("WARNING: Image shape is very large (>1500). Stopping.")
        return
    if min(shape) < 20:
        logger.info("WARNING: Image shape is very small (<20). Stopping.")
        return
    if len(shape) != 3:
        logger.info("WARNING: Image shape is not 3D. Stopping.")
        return

    logger.info("Creating report...")
    ct_input = nib.Nifti1Image(
        np.asanyarray(ct_img.dataobj), ct_img.affine, ct_img.header
    )
    generator = create_aorta_report(
        ct_input,
        args.rois_totalseg,
        args.rois_details,
        metadata,
        tmp_dir,
        logger,
        delete_tmp,
        delete_aux_files,
        args.cpr,
        args.cpr_animated,
        args.aorta_fn,
        args.annulus_fn,
        args.test,
        args.debug,
        args.run_models,
        args.models_parallel,
        f_type,
        "local",
        args.erosion,
        VERSION,
        args.skip_dissection,
    )
    final_result = None
    for result in generator:
        print(f"progress: {result['progress']}, status: {result['status']}")
        if result["progress"] == 100:
            final_result = result
    if final_result is None:
        raise RuntimeError("Aorta report generator did not return a final result.")
    report_image = decompress_and_deserialize(final_result["report_img"])
    nib.save(report_image, args.output_nifti)
    with args.output_json.open("w") as file:
        json.dump(
            convert_to_serializable(final_result["report_json"]),
            file,
            indent=4,
        )
    if args.save_runtime and args.nodeinfo is not None and args.nodeinfo.exists():
        save_runtime(started, "aorta_report", args.nodeinfo)
    if args.cpr_animated is not None:
        os._exit(0)


if __name__ == "__main__":
    main()
