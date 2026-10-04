from __future__ import annotations

import time

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

from totalsegmentator.cropping import crop_to_bbox, crop_to_bbox_nifti, get_bbox_from_mask
from totalsegmentator.postprocessing import keep_largest_blob, remove_small_blobs


def get_bbox_around_center(center, distances, ref_img):
    shape = ref_img.shape
    return [
        [
            max(0, int(center[axis] - distances[axis])),
            min(shape[axis], int(center[axis] + distances[axis])),
        ]
        for axis in range(3)
    ]


def crop_to_masks(
    img_in,
    masks_in,
    addon=(10, 10, 10),
    dtype=np.int32,
    fixed_size=False,
):
    addon = (np.asarray(addon) / img_in.header.get_zooms()[:3]).astype(int)
    reference = masks_in[0]
    combined = np.zeros(reference.shape)
    for mask_img in masks_in:
        combined[mask_img.get_fdata() > 0.5] = 1
    if combined.shape != img_in.shape or not np.allclose(reference.affine, img_in.affine):
        combined = resample_from_to(
            nib.Nifti1Image(combined, reference.affine), img_in, order=0
        ).get_fdata()
    if fixed_size:
        if combined.any():
            coordinates = np.where(combined > 0)
            center = [int(np.mean(coordinates[axis])) for axis in range(3)]
        else:
            print("WARNING: Could not crop because no foreground detected; cropping to image center.")
            center = [size // 2 for size in combined.shape]
        bbox = get_bbox_around_center(center, (addon / 2).astype(int), combined)
    else:
        bbox = get_bbox_from_mask(combined, outside_value=0, addon=addon)
    return crop_to_bbox_nifti(img_in, bbox, dtype=dtype)


def crop_to_point(img_in, center, size=(10, 10, 10), dtype=np.int32):
    radius = np.asarray(size) / 2
    radius = (radius / img_in.header.get_zooms()[:3]).astype(int)
    return crop_to_bbox_nifti(
        img_in, get_bbox_around_center(center, radius, img_in), dtype=dtype
    )


def lazy_load(file_path, target_ref_img, tmp_dir, order=0, dtype=np.float32):
    file_path = file_path if file_path.exists() else (
        file_path.parent / "brachiocephalic_trunc.nii.gz"
        if file_path.name == "brachiocephalic_trunk.nii.gz"
        else file_path
    )
    output_path = tmp_dir / file_path.name
    if output_path.exists():
        print(f"  loading existing ({output_path.name})...")
        image = nib.load(output_path)
        return image, image.get_fdata()
    image = resample_from_to(nib.load(file_path), target_ref_img, order=order)
    data = image.get_fdata()
    if dtype != np.float32:
        data = data.astype(dtype)
    if data.max() <= 1:
        data = keep_largest_blob(data)
    image = nib.Nifti1Image(data, image.affine)
    nib.save(image, output_path)
    return image, data


def crop_to_aorta(ct_path, tmp_dir, logger, totalseg_version="v2"):
    output_path = tmp_dir / "ct_cropped.nii.gz"
    if output_path.exists():
        logger.info("  already cropped (skipping)")
        return output_path
    heart_name = "heart_myocardium" if totalseg_version == "v1" else "heart"
    ct_img = nib.load(ct_path)
    print("Cropping...")
    print(f"  shape Before: {ct_img.shape}")
    ct_img = crop_to_masks(
        ct_img,
        [nib.load(tmp_dir / "aorta.nii.gz"), nib.load(tmp_dir / f"{heart_name}.nii.gz")],
        [20, 20, 20],
        dtype=np.int16,
    )
    print(f"  shape After: {ct_img.shape}")
    nib.save(ct_img, output_path)
    (tmp_dir / "aorta.nii.gz").unlink()
    (tmp_dir / f"{heart_name}.nii.gz").unlink()
    return output_path


__all__ = [
    "crop_to_bbox",
    "crop_to_masks",
    "crop_to_point",
    "crop_to_aorta",
    "get_bbox_around_center",
    "keep_largest_blob",
    "lazy_load",
    "remove_small_blobs",
]
