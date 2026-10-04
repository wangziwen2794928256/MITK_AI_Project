from __future__ import annotations

import nibabel as nib
import numpy as np
from scipy import ndimage
from skimage.measure import label, regionprops
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

from totalsegmentator.aorta_report.centerline import get_distance


def vox_to_mm_space(point, affine):
    return affine[:3, :3] @ point + affine[:3, 3]


def vox_to_mm_space_without_offset(point, affine):
    return affine[:3, :3] @ point


def vox_to_other_vox_space(point, original_affine, target_affine):
    return (np.linalg.inv(target_affine) @ original_affine @ np.append(point, 1))[:3]


def draw_plane(image_shape, normal_vec, point):
    normal_vec = normal_vec / np.linalg.norm(normal_vec)
    coordinates = np.indices(image_shape).reshape(3, -1).T
    coordinates = coordinates[np.abs((point - coordinates) @ normal_vec) < 1]
    image = np.zeros(image_shape)
    image[tuple(coordinates.T)] = 1
    return image


def get_region_diameters(image):
    for region in regionprops(label(image)):
        mask = region.image
        boundary = np.logical_xor(mask, ndimage.binary_erosion(mask))
        points = np.transpose(np.nonzero(boundary))
        distances = pairwise_distances(points)
        first, second = np.unravel_index(np.argmax(distances), distances.shape)
        point_a, point_b = points[first], points[second]
        point_a += np.asarray(region.bbox[: image.ndim])
        point_b += np.asarray(region.bbox[: image.ndim])
        yield distances[first, second], point_a, point_b


def _angle(first_a, first_b, second_a, second_b):
    first = first_b - first_a
    second = second_b - second_a
    return abs(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second) + 1e-7))


def max_perpendicular_line(mask, line_start, line_end):
    boundary = mask - ndimage.binary_erosion(mask)
    points = np.transpose(np.where(boundary == 1))
    candidates = []
    for first_idx in range(len(points)):
        for second_idx in range(first_idx + 1, len(points)):
            first, second = points[first_idx], points[second_idx]
            candidates.append(
                (
                    _angle(first, second, line_start, line_end),
                    np.linalg.norm(second - first),
                    first,
                    second,
                )
            )
    for threshold in (0.1, 0.2, 0.5):
        eligible = [candidate for candidate in candidates if candidate[0] < threshold]
        if eligible:
            best = max(eligible, key=lambda candidate: candidate[1])
            if threshold != 0.1 or best[1] > 5:
                return best[1], best[2], best[3]
    print("WARNING: did not find any perpendicular line. Returning None")
    return 0.0, None, None


def get_region_diameters_pd(mask, spacing, affine, z=1):
    if mask.sum() < 5:
        return np.array(0.0), np.array(0.0), None, None
    _, start, end = next(get_region_diameters(mask))
    diameter_mm = get_distance(start, end, spacing)
    _, start_pd, end_pd = max_perpendicular_line(mask, start, end)
    if start_pd is None:
        return np.array(0.0), np.array(0.0), None, None
    diameter_pd_mm = get_distance(start_pd, end_pd, spacing)
    endpoints = [
        point.tolist() + [int(z)] for point in (start, end, start_pd, end_pd)
    ]
    endpoints_mm = [
        vox_to_mm_space(point, affine).round(4).tolist() for point in endpoints
    ]
    return diameter_mm, diameter_pd_mm, endpoints, endpoints_mm


def find_most_distant_points(mask1, mask2):
    points1, points2 = np.argwhere(mask1), np.argwhere(mask2)
    maximum = 0
    result = (None, None)
    for point1 in points1:
        for point2 in points2:
            distance = np.linalg.norm(point1 - point2)
            if distance > maximum:
                maximum, result = distance, (point1, point2)
    return result


def find_center(mask, integer=False):
    points = np.argwhere(mask)
    if not len(points):
        return None
    center = points.mean(axis=0)
    return np.round(center).astype(np.int32) if integer else center


def _normalize(vector):
    norm = np.linalg.norm(vector)
    return vector if norm == 0 else vector / norm


def _rotation_matrix(source, target):
    source, target = _normalize(source), _normalize(target)
    dot = np.dot(source, target)
    if dot == 1:
        return np.eye(3)
    cross = np.cross(source, target)
    skew = np.array(
        [[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]]
    )
    return np.eye(3) + skew + skew @ skew * (1 - dot) / np.linalg.norm(cross) ** 2


def rotate_affine_of_image_and_translate(
    image, source_vector, target_vector, center, nora_transformation=True, flip_negative=True
):
    if np.dot(source_vector, target_vector) < 0 and flip_negative:
        target_vector = -target_vector
    rotation = np.eye(4)
    rotation[:3, :3] = _rotation_matrix(source_vector, target_vector)
    to_origin, from_origin = np.eye(4), np.eye(4)
    to_origin[:3, 3] -= center
    from_origin[:3, 3] += center
    combined = from_origin @ rotation @ to_origin
    affine = combined @ image.affine if nora_transformation else image.affine @ combined
    return combined, affine


def find_mask_normal(mask):
    points = np.argwhere(mask > 0.5)
    if not points.size:
        raise ValueError("find_mask_normal: mask has no positive voxels")
    if len(points) == 1:
        return np.array([0.0, 0.0, 1.0])
    components = PCA(n_components=min(3, *points.shape)).fit(points - points.mean(axis=0)).components_
    normal = np.asarray(components[-1], dtype=float)
    if len(normal) < 3:
        normal = np.pad(normal, (0, 3 - len(normal)))
    norm = np.linalg.norm(normal)
    return normal / norm if norm > 1e-10 else np.array([0.0, 0.0, 1.0])


def smooth_mask(mask, sigma=1):
    return (ndimage.gaussian_filter(mask.astype(np.float32), sigma=sigma) > 0.5).astype(
        np.uint8
    )
