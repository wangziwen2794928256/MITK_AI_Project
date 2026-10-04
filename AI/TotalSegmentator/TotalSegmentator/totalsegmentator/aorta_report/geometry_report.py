from __future__ import annotations

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to
from scipy import ndimage
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    binary_erosion,
    binary_fill_holes,
)

from totalsegmentator.aorta_report.centerline import (
    get_distance,
    get_index_for_point,
    get_mid_of_points,
    get_normal_for_cl_point,
)
from totalsegmentator.aorta_report.geometry import (
    draw_plane,
    find_center,
    find_mask_normal,
    get_region_diameters,
    max_perpendicular_line,
    rotate_affine_of_image_and_translate,
    vox_to_mm_space,
    vox_to_mm_space_without_offset,
    vox_to_other_vox_space,
)
from totalsegmentator.aorta_report.utils import crop_to_masks, crop_to_point


def keep_closest_blob(data, reference_point, debug=False):
    blob_map, count = ndimage.label(data, structure=np.ones((3, 3, 3)))
    distances = {}
    for blob_idx in range(1, count + 1):
        coordinates = np.asarray(np.where(blob_map == blob_idx)).T
        samples = coordinates[
            np.random.choice(range(len(coordinates)), size=10, replace=True)
        ]
        distances[blob_idx] = np.linalg.norm(reference_point - samples, axis=1).mean()
    if debug:
        print("Blob distances:")
        print(distances)
    if not distances:
        return data
    closest = min(distances, key=distances.get)
    return (blob_map == closest).astype(np.uint8)


def get_plane_aorta_intersection(aorta, plane_point, normal_vec):
    plane = draw_plane(aorta.shape, normal_vec, plane_point)
    intersection = binary_dilation((plane + aorta) > 1, iterations=1)
    return keep_closest_blob(intersection, plane_point)


def get_plane_aorta_intersection_crop(aorta_img, plane_point, normal_vec, crop_size=30):
    cropped = crop_to_point(
        aorta_img, plane_point, [crop_size] * 3, dtype=np.uint8
    )
    cropped_point = vox_to_other_vox_space(
        plane_point, aorta_img.affine, cropped.affine
    )
    intersection = get_plane_aorta_intersection(
        cropped.get_fdata().astype(np.uint8), cropped_point, normal_vec
    )
    return resample_from_to(
        nib.Nifti1Image(intersection.astype(np.uint8), cropped.affine),
        aorta_img,
        order=0,
    )


def _diameter_from_centerline_index(aorta, centerline, index, spacing):
    plane = get_plane_aorta_intersection(
        aorta,
        centerline[index].point,
        get_normal_for_cl_point(centerline, index),
    )
    _, start, end = next(get_region_diameters(plane))
    return get_distance(start, end, spacing)


def get_max_diameter_of_centerline_section(
    aorta_img,
    centerline,
    spacing,
    return_diameter=False,
    subsample=2,
    crop_size=30,
):
    if len(centerline) < 2:
        print(f"WARNING: centerline section too short (len: {len(centerline)})")
        return (0, np.float32(0)) if return_diameter else 0
    areas = {}
    for idx, vertex in enumerate(centerline[::subsample]):
        normal = get_normal_for_cl_point(centerline, idx)
        cropped = crop_to_point(
            aorta_img, vertex.point, [crop_size] * 3, dtype=np.uint8
        )
        point = vox_to_other_vox_space(
            vertex.point, aorta_img.affine, cropped.affine
        )
        intersection = get_plane_aorta_intersection(
            cropped.get_fdata().astype(np.uint8), point, normal
        )
        areas[tuple(vertex.point)] = intersection.sum()
    max_point = max(areas, key=areas.get)
    if not return_diameter:
        return max_point
    index = get_index_for_point(centerline, max_point)
    return max_point, _diameter_from_centerline_index(
        aorta_img.get_fdata().astype(np.uint8), centerline, index, spacing
    )


def get_aorta_section(aorta_img, start, end, centerline, crop_size=30):
    aorta = aorta_img.get_fdata().astype(np.uint8)
    if len(centerline) < 2:
        print("WARNING: centerline section too short to calculate aorta section")
        return nib.Nifti1Image(np.zeros_like(aorta), aorta_img.affine)
    start_intersection = get_plane_aorta_intersection_crop(
        aorta_img,
        centerline[start].point,
        get_normal_for_cl_point(centerline, start),
        crop_size,
    ).get_fdata()
    end_intersection = get_plane_aorta_intersection_crop(
        aorta_img,
        centerline[end].point,
        get_normal_for_cl_point(centerline, end),
        crop_size,
    ).get_fdata()
    midpoint = centerline[get_mid_of_points(centerline, start, end)].point
    aorta[start_intersection > 0] = 0
    aorta[end_intersection > 0] = 0
    blobs, _ = ndimage.label(aorta)
    selected = blobs[tuple(np.asarray(midpoint, dtype=int))]
    return nib.Nifti1Image((blobs == selected).astype(np.uint8), aorta_img.affine)


def get_2d_plane(mask, ct_img, mask_2, mask_3, order=0, dtype=np.uint8, addon=(0, 0, 0)):
    mask_cropped = crop_to_masks(mask, [mask], addon, dtype=np.uint8, fixed_size=True)
    ct_cropped = crop_to_masks(ct_img, [mask], addon, dtype=dtype, fixed_size=True)
    mask_data = mask_cropped.get_fdata()
    mask_2_cropped = (
        crop_to_masks(mask_2, [mask], addon, dtype=np.uint8, fixed_size=True)
        if mask_2 is not None
        else None
    )
    mask_3_cropped = (
        crop_to_masks(mask_3, [mask], addon, dtype=np.uint8, fixed_size=True)
        if mask_3 is not None
        else None
    )
    center = vox_to_mm_space(find_center(mask_data), mask_cropped.affine)
    normal = vox_to_mm_space_without_offset(
        find_mask_normal(mask_data), mask_cropped.affine
    )
    target = vox_to_mm_space_without_offset(
        np.array([0, 0, 1]), mask_cropped.affine
    )
    rotation, affine = rotate_affine_of_image_and_translate(
        mask_cropped, normal, target, center, nora_transformation=True
    )
    inverse_affine = np.linalg.inv(rotation) @ mask_cropped.affine

    def rotate(image, interpolation):
        if image is None:
            return None
        return resample_from_to(
            nib.Nifti1Image(image.get_fdata(), affine),
            mask_cropped,
            order=interpolation,
        )

    return (
        rotate(mask_cropped, 0),
        rotate(ct_cropped, order),
        rotate(mask_2_cropped, 0),
        rotate(mask_3_cropped, 0),
        inverse_affine,
    )


def process_plane(ct_img, mask_img, mask_img_2, output_path, smooth=20):
    from totalsegmentator.aorta_report.plotting import plot_img

    image = ct_img.get_fdata()
    mask = binary_fill_holes(binary_closing(mask_img.get_fdata())).astype(np.uint8)
    mask_2 = (
        binary_fill_holes(binary_closing(mask_img_2.get_fdata())).astype(np.uint8)
        if mask_img_2 is not None
        else None
    )
    center = find_center(mask, True)
    image, mask = image[:, :, center[2]], mask[:, :, center[2]]
    if mask_2 is not None:
        mask_2 = mask_2[:, :, center[2]]
    mask = binary_fill_holes(binary_closing(mask)).astype(np.uint8)
    spacing = np.asarray(ct_img.header.get_zooms())[:2]
    _, start, end = next(get_region_diameters(mask))
    _, start_pd, end_pd = max_perpendicular_line(mask, start, end)
    figure = plot_img(
        image,
        mask,
        mask_2,
        [start, end, start_pd, end_pd],
        vmin=-700,
        vmax=1000,
        smooth=smooth,
    )
    figure.savefig(output_path, bbox_inches="tight", pad_inches=0)
