from __future__ import annotations

import asyncio
import gc
import shutil
import tempfile
import time
from collections import defaultdict
from importlib import resources
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
)
from tqdm import tqdm

from totalsegmentator.aorta_report.centerline import (
    add_point_to_centerline,
    extend_centerline,
    get_centerline,
    get_closest_centerline_section,
    get_closest_point_of_centerline,
    get_closest_point_of_centerline_by_plane,
    get_distance,
    get_index_for_point,
    get_len_path_mm,
    get_mid_of_points,
    get_next_point_by_distance,
    get_normal_for_cl_point,
    reorder_centerline,
    resample_cl_to_same_distance,
)
from totalsegmentator.aorta_report.cpr import cpr
from totalsegmentator.aorta_report.cpr_animated import generate_animated_cpr_nifti
from totalsegmentator.aorta_report.geometry import (
    find_center,
    find_mask_normal,
    find_most_distant_points,
    get_region_diameters,
    get_region_diameters_pd,
    smooth_mask,
)
from totalsegmentator.aorta_report.geometry_report import (
    get_2d_plane,
    get_aorta_section,
    get_max_diameter_of_centerline_section,
    get_plane_aorta_intersection,
    process_plane,
)
from totalsegmentator.aorta_report.model_runner import (
    get_aorta_fast,
    get_contrast_phase,
    run_models_consecutive,
    run_models_parallel,
)
from totalsegmentator.aorta_report.nifti import combine_as_nifti
from totalsegmentator.aorta_report.plotting import (
    plot_aorta_3d,
    plot_cpr_overview,
    plot_masks_3d,
)
from totalsegmentator.aorta_report.utils import (
    crop_to_aorta,
    crop_to_bbox,
    crop_to_masks,
    get_bbox_around_center,
    keep_largest_blob,
    lazy_load,
    remove_small_blobs,
)
from totalsegmentator.dicom_io import dcm_to_nifti
from totalsegmentator.resampling import change_spacing
from totalsegmentator.reporting import generate_html, setup_logger
from totalsegmentator.serialization_utils import filestream_to_nifti, serialize_and_compress


def _diameter_profiles(
    mask_data,
    centerline,
    spacing,
    crop_size_mm,
    subsample=5,
    target_points=70,
):
    if not centerline:
        return np.empty(0), [np.empty(0) for _ in mask_data]
    radius = np.maximum(
        1, (np.asarray([crop_size_mm] * 3) / 2 / np.asarray(spacing)).astype(int)
    )
    stride = max(subsample, int(np.ceil(len(centerline) / max(target_points, 1))))
    indexes = list(range(0, len(centerline), stride))
    if indexes[-1] != len(centerline) - 1:
        indexes.append(len(centerline) - 1)
    cumulative = np.zeros(len(centerline))
    for idx in range(1, len(centerline)):
        cumulative[idx] = cumulative[idx - 1] + get_distance(
            centerline[idx].point, centerline[idx - 1].point, spacing
        )
    result = [np.full(len(indexes), np.nan) for _ in mask_data]
    has_data = [np.count_nonzero(mask) >= 5 for mask in mask_data]
    for output_idx, centerline_idx in enumerate(
        tqdm(indexes, desc="Diameter profiles")
    ):
        point = np.asarray(centerline[centerline_idx].point)
        bbox = get_bbox_around_center(point, radius, mask_data[0])
        cropped_point = point - np.asarray([axis[0] for axis in bbox])
        normal = get_normal_for_cl_point(centerline, centerline_idx)
        for mask_idx, mask in enumerate(mask_data):
            if not has_data[mask_idx]:
                continue
            intersection = get_plane_aorta_intersection(
                crop_to_bbox(mask, bbox).astype(np.uint8), cropped_point, normal
            )
            try:
                _, start, end = next(get_region_diameters(intersection))
                result[mask_idx][output_idx] = get_distance(start, end, spacing)
            except StopIteration:
                pass
    return cumulative[-1] - cumulative[indexes], result


def create_aorta_report(
    ct_bytes,
    rois_totalseg_dir,
    rois_details_dir,
    metadata,
    tmp_dir,
    logger,
    delete_tmp=True,
    delete_aux_files=True,
    cpr_path=None,
    cpr_animated_path=None,
    aorta_fn="aorta.nii.gz",
    annulus_fn="annulus.nii.gz",
    test="None",
    debug=False,
    run_models=False,
    models_parallel=False,
    f_type="nii",
    host="local",
    erosion=False,
    version="0.1.1",
    skip_dissection=False,
):
    """Generate an aorta report and yield source-compatible progress dictionaries."""
    yield {"id": 1, "progress": 2, "status": "Loading code"}
    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp())
        delete_tmp = delete_aux_files = True
    tmp_dir = Path(tmp_dir)
    model_dir = tmp_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    if logger is None:
        logger = setup_logger(
            tmp_dir / "log.txt", name=f"totalseg_aorta_report.{tmp_dir}"
        )

    ct_path = model_dir / "ct.nii.gz"
    if isinstance(ct_bytes, Path):
        ct_path = ct_bytes
    elif isinstance(ct_bytes, nib.Nifti1Image):
        nib.save(ct_bytes, ct_path)
    elif f_type == "dicom":
        if ct_path.exists():
            logger.info("Found existing nifti files. Skipping conversion...")
        else:
            logger.info("Converting dicom to nifti...")
            dcm_to_nifti(ct_bytes, ct_path, tmp_dir=tmp_dir, verbose=True)
    else:
        nib.save(filestream_to_nifti(ct_bytes, gzipped=f_type == "niigz"), ct_path)

    if run_models:
        yield {"id": 2, "progress": 5, "status": "Running models"}
        get_aorta_fast(ct_path, model_dir, logger, "v2", host)
        ct_path = crop_to_aorta(ct_path, model_dir, logger, "v2")
        new_spacing = 1.5 if test == "basic" else 0.8
        ct_img = change_spacing(
            nib.load(ct_path), new_spacing, order=3, dtype=np.int16
        )
        nib.save(ct_img, ct_path)
        print(f"Resampled CT to {new_spacing}mm: {ct_img.shape}")
        if models_parallel:
            asyncio.run(run_models_parallel(ct_path, model_dir, logger, host))
        else:
            for idx, status in enumerate(
                run_models_consecutive(ct_path, model_dir, logger)
            ):
                yield {
                    "id": 3,
                    "progress": 10 + idx * 2,
                    "status": status,
                }
        rois_totalseg_dir = rois_details_dir = model_dir

    rois_totalseg_dir, rois_details_dir = (
        Path(rois_totalseg_dir),
        Path(rois_details_dir),
    )
    yield {
        "id": 3,
        "progress": 19,
        "status": "Running contrast phase detection",
    }
    contrast_phase = get_contrast_phase(ct_path, model_dir, logger, host)
    logger.info(f"Contrast phase: {contrast_phase}")

    vessels_plot = [
        "brachiocephalic_trunk",
        "subclavian_artery_left",
        "common_carotid_artery_left",
        "renal_arteries",
        "celiac_trunk",
        "superior_mesenteric_artery",
        "iliac_artery_left",
        "iliac_artery_right",
    ]
    structures = {
        "brachio": ["brachiocephalic_trunk", "roi"],
        "subclavian": ["subclavian_artery_left", "roi"],
        "celiac": ["celiac_trunk", "roi"],
        "T12": ["vertebrae_T12", "roi"],
        "sinotub_junc": ["sinotubular_junction", "roi_cropped_aorta"],
        "iliac": ["iliac_artery_right", "roi"],
    }
    structures = {
        key: {
            "name": value[0],
            "data": None,
            "endpoint": None,
            "cl_point": None,
            "cl_idx": None,
            "cl_normal": None,
            "roi_dir": value[1],
            "empty": False,
        }
        for key, value in structures.items()
    }

    yield {"id": 4, "progress": 20, "status": "Loading data"}
    annulus_img = nib.as_closest_canonical(nib.load(rois_details_dir / annulus_fn))
    if annulus_img.affine[:3, :3].min() < 0:
        logger.info("WARNING: Affine contains negative values!")
    aorta_crop = nib.as_closest_canonical(
        nib.load(rois_totalseg_dir / aorta_fn)
    )
    aorta_crop = nib.Nifti1Image(
        keep_largest_blob(aorta_crop.get_fdata()), aorta_crop.affine
    )
    annulus_img = crop_to_masks(
        annulus_img, [aorta_crop], [20, 20, 20], dtype=np.uint8
    )
    annulus_img = change_spacing(annulus_img, 0.8, order=0)
    annulus = annulus_img.get_fdata()
    annulus_volume = annulus.sum() * np.prod(annulus_img.header.get_zooms())
    if annulus_volume < 200:
        logger.info("WARNING: Annulus is empty!")
    if debug:
        nib.save(annulus_img, tmp_dir / "annulus.nii.gz")

    affine, spacing = annulus_img.affine, annulus_img.header.get_zooms()
    ct_img, ct = lazy_load(ct_path, annulus_img, tmp_dir, order=3)
    aorta_img, aorta = lazy_load(
        rois_totalseg_dir / aorta_fn, annulus_img, tmp_dir, order=0
    )
    aorta_totalseg = binary_fill_holes(aorta).astype(np.uint8)
    true_path = rois_details_dir / "aorta_true_lumen.nii.gz"
    calculate_lumen = true_path.exists() and not skip_dissection and contrast_phase != "native"
    if not true_path.exists():
        logger.info(
            "WARNING: No true/false lumen found! Using aorta + empty array instead."
        )
    elif skip_dissection:
        logger.info(
            "WARNING: skip_dissection=True! Using aorta + empty array instead."
        )
    elif contrast_phase == "native":
        logger.info(
            "WARNING: contrast phase is native. Not using true/false lumen segmentation!"
        )
    if calculate_lumen:
        _, true_lumen = lazy_load(true_path, annulus_img, tmp_dir, order=3)
        _, false_lumen = lazy_load(
            rois_details_dir / "aorta_false_lumen.nii.gz",
            annulus_img,
            tmp_dir,
            order=3,
        )
    else:
        true_lumen, false_lumen = aorta.copy(), np.zeros_like(aorta)
    false_lumen = remove_small_blobs(false_lumen, [20, 1e9])
    aorta[true_lumen > 0.5] = 1
    aorta[false_lumen > 0.5] = 1
    aorta = binary_fill_holes(aorta).astype(np.uint8)
    true_lumen = aorta.copy()
    true_lumen[false_lumen > 0.5] = 0
    annulus = binary_closing(annulus, iterations=1)
    annulus_center = find_center(annulus)

    for name, structure in structures.items():
        directory = rois_details_dir if name == "sinotub_junc" else rois_totalseg_dir
        _, data = lazy_load(
            directory / f"{structure['name']}.nii.gz",
            annulus_img,
            tmp_dir,
            order=0,
        )
        if data.sum() == 0:
            logger.info(f"WARNING: {name} is empty!")
        structure["data"] = data
    all_vessels = np.zeros_like(aorta)
    for vessel in vessels_plot:
        _, data = lazy_load(
            rois_totalseg_dir / f"{vessel}.nii.gz",
            annulus_img,
            tmp_dir,
            order=0,
        )
        all_vessels[data > 0.5] = 1

    yield {"id": 5, "progress": 25, "status": "Creating centerline"}
    started = time.time()
    aorta_smooth = smooth_mask(
        binary_dilation(aorta, iterations=4), sigma=6
    )
    centerline_image, centerline = get_centerline(aorta_smooth, debug=debug)
    centerline = reorder_centerline(centerline)
    centerline_image, centerline = add_point_to_centerline(
        centerline_image, centerline, annulus_center
    )
    extended_image, extended = extend_centerline(
        centerline_image, centerline, 1
    )
    if annulus[tuple(centerline[-1].point.astype(int))]:
        logger.info("  New annulus point is in annulus mask!")
        centerline_image, centerline = extended_image, extended
    else:
        logger.info(
            "WARNING: New annulus point is not in annulus mask! Skipping the extension."
        )
    if get_len_path_mm(centerline, spacing) < 2:
        logger.info(
            "WARNING: Centerline is very short (<2mm). This may indicate issues with the input data."
        )
    centerline_resampled = resample_cl_to_same_distance(
        centerline, affine, 1
    )
    logger.info(f"  get_centerline took {time.time() - started:.2f}s")

    for name, structure in structures.items():
        if name in ("T12", "sinotub_junc", "iliac"):
            intersection = (
                structure["data"]
                if structure["data"].sum() * np.prod(spacing) >= 200
                else np.zeros_like(structure["data"])
            )
        else:
            intersection = (
                binary_dilation(structure["data"], iterations=1) + aorta
            ) > 1
        structure["intersect_aorta"] = intersection
        if intersection.sum() == 0:
            logger.info(
                f"WARNING: {name} is (almost) empty or has no intersection with aorta! Setting to empty."
            )
            structure["empty"] = True
    if structures["brachio"]["empty"] or structures["subclavian"]["empty"]:
        structures["brachio"]["empty"] = structures["subclavian"]["empty"] = True
        logger.info(
            "WARNING: Setting subclavian+brachio to empty because one or both are empty."
        )
        brachio_point = subclavian_point = None
    else:
        brachio_point, subclavian_point = find_most_distant_points(
            structures["brachio"]["intersect_aorta"],
            structures["subclavian"]["intersect_aorta"],
        )
    for name, structure in structures.items():
        if structure["empty"]:
            continue
        intersection = structure["intersect_aorta"]
        if name in ("celiac", "iliac"):
            indexes = np.where(intersection > 0)
            top = np.argmax(indexes[2])
            endpoint = tuple(indexes[axis][top] for axis in range(3))
        elif name == "brachio":
            endpoint = tuple(brachio_point)
        elif name == "subclavian":
            endpoint = tuple(subclavian_point)
        else:
            endpoint = tuple(find_center(intersection, True))
        structure["endpoint"] = endpoint
        segment = (
            centerline[structures["brachio"]["cl_idx"] :]
            if name == "sinotub_junc"
            else (
                get_closest_centerline_section(endpoint, centerline, 40, spacing[2])
                if name == "T12"
                else centerline
            )
        )
        get_closest_point_of_centerline_by_plane(endpoint, segment)
        point = get_closest_point_of_centerline(endpoint, segment)
        structure["cl_point"] = point
        structure["cl_idx"] = get_index_for_point(centerline, point)
        structure["cl_normal"] = get_normal_for_cl_point(
            centerline, structure["cl_idx"]
        )

    landmarks = defaultdict(dict)
    dependencies = {
        1: ["annulus"],
        2: ["annulus", "sinotub_junc"],
        3: ["sinotub_junc"],
        4: ["sinotub_junc", "brachio"],
        5: ["brachio"],
        6: ["brachio", "subclavian"],
        7: ["subclavian"],
        8: ["subclavian", "T12"],
        9: ["T12"],
        10: ["celiac"],
        11: ["iliac"],
    }
    structures["annulus"] = {"empty": annulus_volume < 200}
    for number, required in dependencies.items():
        landmarks[number]["depends_on"] = required
        landmarks[number]["empty"] = any(
            structures[name]["empty"] for name in required
        )
        for name in required:
            if structures[name]["empty"]:
                logger.info(
                    f"WARNING: landmark {number} is empty because {name} is empty!"
                )
    definitions = {
        1: "annulus",
        2: "sinuses of valsalva",
        3: "sinotub. junc.",
        4: "mid asc. aorta",
        5: "distal asc. aorta",
        6: "mid aortic arch",
        7: "proximal desc. aorta",
        8: "mid desc. aorta",
        9: "desc. aorta (T12)",
        10: "abd. aorta (celiac artery)",
        11: "abd. aorta (bifurcation)",
    }
    for number, name in definitions.items():
        landmarks[number]["name"] = name
    if not landmarks[11]["empty"]:
        landmarks[11]["cl_idx"] = get_next_point_by_distance(
            centerline, 0, 20, spacing
        )
    if not landmarks[10]["empty"]:
        landmarks[10]["cl_idx"] = structures["celiac"]["cl_idx"]
    if not landmarks[9]["empty"]:
        if (
            not structures["celiac"]["empty"]
            and structures["T12"]["cl_idx"] < structures["celiac"]["cl_idx"]
        ):
            logger.info(
                "WARNING: T12 is lower than celiac! Setting to celiac + 10mm."
            )
            landmarks[9]["cl_idx"] = get_next_point_by_distance(
                centerline, structures["celiac"]["cl_idx"], 10, spacing
            )
        else:
            landmarks[9]["cl_idx"] = structures["T12"]["cl_idx"]
    if not landmarks[7]["empty"]:
        landmarks[7]["cl_idx"] = get_next_point_by_distance(
            centerline, structures["subclavian"]["cl_idx"], -5, spacing
        )
    if not landmarks[8]["empty"]:
        landmarks[8]["cl_idx"] = get_mid_of_points(
            centerline, landmarks[7]["cl_idx"], landmarks[9]["cl_idx"]
        )
    if not landmarks[5]["empty"]:
        landmarks[5]["cl_idx"] = get_next_point_by_distance(
            centerline, structures["brachio"]["cl_idx"], 5, spacing
        )
    if not landmarks[6]["empty"]:
        landmarks[6]["cl_idx"] = get_mid_of_points(
            centerline, landmarks[5]["cl_idx"], landmarks[7]["cl_idx"]
        )
    if not landmarks[3]["empty"]:
        landmarks[3]["cl_idx"] = structures["sinotub_junc"]["cl_idx"]
    if not landmarks[4]["empty"]:
        landmarks[4]["cl_idx"] = get_mid_of_points(
            centerline, landmarks[3]["cl_idx"], landmarks[5]["cl_idx"]
        )
    if not landmarks[1]["empty"]:
        landmarks[1]["cl_idx"] = get_next_point_by_distance(
            centerline, len(centerline) - 1, 0, spacing
        )
    if not landmarks[2]["empty"]:
        start = get_next_point_by_distance(
            centerline, landmarks[3]["cl_idx"], 10, spacing
        )
        end = get_next_point_by_distance(
            centerline, landmarks[1]["cl_idx"], -10, spacing
        )
        if start is not None and end is not None and start < end:
            point = get_max_diameter_of_centerline_section(
                aorta_img, centerline[start:end], spacing, subsample=1, crop_size=80
            )
            landmarks[2]["cl_idx"] = get_index_for_point(centerline, point)
        else:
            print(
                "WARNING: Distance between annulus sinutub. jun. is too small! "
                "As robust backup take middle as sinuses of valsalva."
            )
            landmarks[2]["cl_idx"] = get_mid_of_points(
                centerline, landmarks[1]["cl_idx"], landmarks[3]["cl_idx"]
            )

    yield {"id": 6, "progress": 35, "status": "Getting max diameter"}
    for number, landmark in landmarks.items():
        if landmark["empty"]:
            landmark["diameter_tmp"] = 0
            continue
        point = centerline[landmark["cl_idx"]].point
        if landmark["name"] == "annulus":
            normal = find_mask_normal(annulus.astype(np.uint8))
            plane = get_plane_aorta_intersection(
                annulus.astype(np.uint8), point, normal
            ).astype(np.uint8)
            if erosion:
                plane = binary_erosion(plane, iterations=1).astype(np.uint8)
        else:
            normal = get_normal_for_cl_point(centerline, landmark["cl_idx"])
            source = binary_erosion(aorta, iterations=1) if erosion else aorta
            plane = get_plane_aorta_intersection(source, point, normal).astype(
                np.uint8
            )
        landmark["cl_point"], landmark["roi"] = point, plane
        try:
            _, start, end = next(get_region_diameters(plane))
            landmark["diameter_tmp"] = get_distance(start, end, spacing)
        except StopIteration:
            logger.info(
                f"WARNING: for landmark {number} no diameter was found!"
            )
            landmark["diameter_tmp"] = 0
    max_diameter = max(item["diameter_tmp"] for item in landmarks.values())

    for number, landmark in landmarks.items():
        logger.info(f"  Processing landmark {number}")
        if landmark["empty"]:
            for metric in (
                "area",
                "area_fl",
                "area_tl",
                "diameter",
                "diameter_fl",
                "diameter_tl",
                "diameter_pd",
                "diameter_fl_pd",
                "diameter_tl_pd",
                "start",
                "end",
                "start_pd",
                "end_pd",
            ):
                landmark[metric] = None
            continue
        plane_img = nib.Nifti1Image(landmark["roi"], affine)
        false_plane = nib.Nifti1Image(
            np.logical_and(landmark["roi"], false_lumen).astype(np.uint8),
            affine,
        )
        true_plane = nib.Nifti1Image(
            np.logical_and(landmark["roi"], true_lumen).astype(np.uint8),
            affine,
        )
        plane_2d, ct_2d, false_2d, true_2d, inverse_affine = get_2d_plane(
            plane_img,
            ct_img,
            false_plane,
            true_plane,
            order=3,
            dtype=np.int32,
            addon=[max_diameter * 1.4] * 3,
        )
        image_path = tmp_dir / f"plane_{number}.png"
        process_plane(ct_2d, plane_2d, false_2d, image_path)
        landmark["plane_img"] = str(image_path)
        plane_data = plane_2d.get_fdata().astype(np.uint8)
        false_data = false_2d.get_fdata().astype(np.uint8)
        true_data = true_2d.get_fdata().astype(np.uint8)
        center = find_center(plane_data, True)
        slice_spacing = np.asarray(plane_2d.header.get_zooms())[:2]
        plane_data = binary_fill_holes(
            binary_closing(plane_data[:, :, center[2]])
        ).astype(np.uint8)
        false_data = binary_fill_holes(
            binary_closing(false_data[:, :, center[2]])
        ).astype(np.uint8)
        true_data = binary_fill_holes(
            binary_closing(true_data[:, :, center[2]])
        ).astype(np.uint8)
        landmark["area"] = (
            plane_data.sum() * np.prod(slice_spacing) / 100
        ).round(1)
        landmark["area_fl"] = (
            false_data.sum() * np.prod(slice_spacing) / 100
        ).round(1)
        landmark["area_tl"] = (
            true_data.sum() * np.prod(slice_spacing) / 100
        ).round(1)
        diameter, perpendicular, endpoints, endpoints_mm = get_region_diameters_pd(
            plane_data, slice_spacing, inverse_affine, z=center[2]
        )
        landmark["diameter"] = (diameter / 10).round(1)
        landmark["diameter_pd"] = (perpendicular / 10).round(1)
        endpoint_values = endpoints_mm or [None] * 4
        (
            landmark["start"],
            landmark["end"],
            landmark["start_pd"],
            landmark["end_pd"],
        ) = endpoint_values
        voxel_values = endpoints or [None] * 4
        (
            landmark["start_voxel"],
            landmark["end_voxel"],
            landmark["start_pd_voxel"],
            landmark["end_pd_voxel"],
        ) = voxel_values
        false_diameter, false_pd, _, _ = get_region_diameters_pd(
            false_data, slice_spacing, inverse_affine
        )
        true_diameter, true_pd, _, _ = get_region_diameters_pd(
            true_data, slice_spacing, inverse_affine
        )
        landmark["diameter_fl"] = (false_diameter / 10).round(1)
        landmark["diameter_fl_pd"] = (false_pd / 10).round(1)
        landmark["diameter_tl"] = (true_diameter / 10).round(1)
        landmark["diameter_tl_pd"] = (true_pd / 10).round(1)

    sections = {
        "aorta_ascending": [1, 5],
        "aorta_arch": [5, 7],
        "aorta_descending": [7, 9],
        "aorta_abdominal": [9, 11],
        "aorta_total": [1, 11],
    }
    section_stats, section_masks = {}, []
    for name, (start_number, end_number) in sections.items():
        logger.info(f"  Processing section: {name}...")
        start_landmark, end_landmark = (
            landmarks[start_number],
            landmarks[end_number],
        )
        if start_landmark["empty"] or end_landmark["empty"]:
            logger.info(f"WARNING: {name} is empty!")
            section_stats[name] = {
                key: None
                for key in (
                    "length",
                    "max_diameter",
                    "max_diameter_perpendicular",
                    "volume",
                    "volume_true_lumen",
                    "volume_false_lumen",
                )
            }
            continue
        _, section_diameter = get_max_diameter_of_centerline_section(
            aorta_img,
            centerline[end_landmark["cl_idx"] : start_landmark["cl_idx"]],
            spacing,
            True,
            subsample=1,
            crop_size=max_diameter * 1.4,
        )
        if section_diameter > max_diameter:
            logger.info(
                f"WARNING: max_dia of aorta section {name} is larger than max_dia from "
                f"landmarks ({section_diameter:.2f} > {max_diameter:.2f})"
            )
        if name == "aorta_total":
            section = aorta_totalseg
        else:
            section_img = get_aorta_section(
                aorta_img,
                start_landmark["cl_idx"],
                max(0, end_landmark["cl_idx"] - 2),
                centerline,
                crop_size=max_diameter * 1.4,
            )
            section = section_img.get_fdata()
            section_masks.append(section)
        true_section, false_section = section * true_lumen, section * false_lumen
        section_stats[name] = {
            "length": (
                get_len_path_mm(
                    centerline[
                        end_landmark["cl_idx"] : start_landmark["cl_idx"]
                    ],
                    spacing,
                )
                / 10
            ).round(1),
            "max_diameter": (section_diameter / 10).round(1),
            "volume": (section.sum() * np.prod(spacing) / 1000).round(1),
            "volume_true_lumen": (
                true_section.sum() * np.prod(spacing) / 1000
            ).round(1),
            "volume_false_lumen": (
                false_section.sum() * np.prod(spacing) / 1000
            ).round(1),
        }

    if cpr_path is not None or cpr_animated_path is not None:
        logger.info("Calculating CPR...")
        cpr_started = time.time()
        cpr_values = cpr(
            ct_img,
            aorta_img,
            centerline,
            max_diameter,
            fast=debug,
            debug=debug,
            extra_mask_imgs=[
                nib.Nifti1Image(true_lumen.astype(np.uint8), affine),
                nib.Nifti1Image(false_lumen.astype(np.uint8), affine),
            ],
            return_info=True,
        )
        res_ct, res_seg, lumen_images, cpr_info = cpr_values
        true_cpr, false_cpr = lumen_images
        logger.info(f"CPR took {time.time() - cpr_started:.2f}s")
        logger.info("Calculating diameter profile on original masks...")
        profile_started = time.time()
        curve_positions, profiles = _diameter_profiles(
            [aorta_totalseg, true_lumen, false_lumen],
            centerline_resampled,
            spacing,
            max(max_diameter * 1.6, 60),
        )
        aorta_curve, true_curve, false_curve = profiles
        logger.info(f"Diameter profile took {time.time() - profile_started:.2f}s")
        if cpr_path is not None:
            plot_cpr_overview(
                res_ct,
                res_seg,
                cpr_path,
                -700,
                1000,
                true_cpr,
                false_cpr,
                landmarks,
                np.asarray([vertex.point for vertex in centerline]),
                affine,
                cpr_info["sample_positions_mm"],
                curve_positions,
                aorta_curve,
                true_curve,
                false_curve,
                metadata,
                tmp_dir,
            )

    frames, smoothing = 24, 100
    yield {
        "id": 7,
        "progress": 85,
        "status": "Generating report images",
    }
    from xvfbwrapper import Xvfb

    with Xvfb():
        plot_aorta_3d(
            aorta,
            true_lumen,
            false_lumen,
            all_vessels,
            centerline_image,
            landmarks,
            tmp_dir,
            smoothing,
            frames,
            debug,
        )
        plot_masks_3d(
            [true_lumen, false_lumen],
            tmp_dir,
            "preview_tf_lumen_",
            smoothing,
            frames,
            debug,
            ["green", "red"],
        )
        if cpr_animated_path is not None:
            generate_animated_cpr_nifti(
                aorta,
                centerline,
                affine,
                res_ct,
                res_seg,
                true_cpr,
                false_cpr,
                cpr_info,
                curve_positions,
                aorta_curve,
                true_curve,
                false_curve,
                landmarks,
                max_diameter,
                cpr_animated_path,
                tmp_dir,
                smoothing=smoothing,
                logger=logger,
                metadata=metadata,
            )
        gc.collect()

    landmarks = dict(sorted(landmarks.items()))
    yield {
        "id": 8,
        "progress": 90,
        "status": "Generating report html",
    }
    assets_dir = resources.files("totalsegmentator").joinpath(
        "resources/aorta_report"
    )
    plot_types = ["preview_3d_rotating_", "preview_tf_lumen_"]
    for plot_idx, plot_type in enumerate(plot_types):
        for frame_idx in range(frames):
            generate_html(
                str(assets_dir),
                "report_template_frontpage.html",
                {
                    "landmarks": landmarks,
                    "section_stats": section_stats,
                    "preview_3d": str(
                        tmp_dir / f"{plot_type}{frame_idx:06d}.png"
                    ),
                    "metadata": metadata,
                },
                tmp_dir
                / f"aorta_report_frontpage_{plot_idx * frames + frame_idx:02d}.png",
                width=900,
                png_logo_hosts=("rigi", "rndapollolp01.uhbs.ch"),
            )
    slices = [
        tmp_dir / f"aorta_report_frontpage_{idx:02d}.png"
        for idx in range(len(plot_types) * frames)
    ]
    report_image = combine_as_nifti(slices)
    for landmark in landmarks.values():
        if not landmark["empty"]:
            for key in ("roi", "plane_img", "cl_idx", "cl_point"):
                del landmark[key]
        for key in (
            "diameter_tmp",
            "color",
            "txt_offset",
            "empty",
            "depends_on",
        ):
            del landmark[key]
    results = {
        "landmarks": landmarks,
        "section_stats": section_stats,
        "metadata": metadata,
    }
    yield {
        "id": 9,
        "progress": 95,
        "status": "Combine masks for output",
    }
    masks = {}
    for name in (
        aorta_fn.split(".")[0],
        "aorta_false_lumen",
        "aorta_true_lumen",
    ):
        path = tmp_dir / f"{name}.nii.gz"
        if path.exists():
            image = nib.load(path)
            masks[name] = nib.Nifti1Image(
                image.get_fdata().astype(np.uint8), image.affine, image.header
            )
        else:
            logger.info(
                f"WARNING: Mask {name} not found for final output. Skipping."
            )
    serialized_report = serialize_and_compress(report_image)
    serialized_masks = serialize_and_compress(masks)
    if delete_aux_files:
        for path in tmp_dir.glob("*.png"):
            if "cpr.png" not in str(path):
                path.unlink()
        for path in tmp_dir.glob("*.html"):
            path.unlink()
    if delete_tmp:
        shutil.rmtree(tmp_dir)
    yield {"id": 10, "progress": 96, "status": "Returning data"}
    yield {
        "id": 11,
        "progress": 100,
        "status": "Done",
        "report_img": serialized_report,
        "report_json": results,
        "masks": serialized_masks,
    }
