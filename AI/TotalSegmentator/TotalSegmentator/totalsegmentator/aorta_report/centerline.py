from __future__ import annotations

from collections import deque
from itertools import combinations

import networkx as nx
import nibabel as nib
import numpy as np
from skimage.morphology import skeletonize


class Vertex:
    def __init__(self, point):
        self.point = np.asarray(point)

    def __str__(self):
        return str(self.point)


_NEIGHBORS = np.array(
    [
        (x, y, z)
        for z in (-1, 0, 1)
        for x in (-1, 0, 1)
        for y in (-1, 0, 1)
        if (x, y, z) != (0, 0, 0)
    ]
)


def _build_trees(data):
    """Build the same breadth-first spanning trees used by the source report."""
    unseen = data.astype(bool).copy()
    graphs = []
    shape = np.asarray(unseen.shape)
    while unseen.any():
        start = np.argwhere(unseen)[0]
        start_vertex = Vertex(start)
        graph = nx.Graph()
        graph.add_node(start_vertex)
        unseen[tuple(start)] = False
        queue = deque([start_vertex])
        while queue:
            current = queue.popleft()
            for offset in _NEIGHBORS:
                point = current.point + offset
                if np.any(point < 0) or np.any(point >= shape) or not unseen[tuple(point)]:
                    continue
                vertex = Vertex(point)
                graph.add_edge(current, vertex)
                unseen[tuple(point)] = False
                queue.append(vertex)
        graphs.append(graph)
    return graphs


def get_len_path(path):
    return sum(
        np.linalg.norm(path[idx].point - path[idx - 1].point)
        for idx in range(1, len(path))
    )


def get_distance(a, b, spacing):
    return np.sqrt(np.sum(((np.asarray(a) - np.asarray(b)) * spacing) ** 2))


def get_len_path_mm(path, spacing):
    return np.float32(
        sum(
            get_distance(path[idx].point, path[idx - 1].point, spacing)
            for idx in range(1, len(path))
        )
    )


def get_centerline(data, debug=False):
    if debug:
        print("Skeletonize...")
    skeleton = skeletonize(data.copy() > 0)
    if debug:
        print("Building graph....")
    candidates = []
    for graph in _build_trees(skeleton):
        endpoints = [node for node in graph.nodes if graph.degree(node) == 1]
        if len(endpoints) < 2:
            continue
        longest = max(
            combinations(endpoints, 2),
            key=lambda pair: nx.shortest_path_length(graph, pair[0], pair[1]),
        )
        candidates.append(nx.shortest_path(graph, longest[0], longest[1]))
    if not candidates:
        raise ValueError("Unable to create a centerline from the aorta mask.")
    path = max(candidates, key=get_len_path)
    image = np.zeros(skeleton.shape)
    for vertex in path:
        image[tuple(vertex.point)] = 1
    return image, path


def reorder_centerline(centerline):
    return centerline[::-1] if centerline[0].point[2] > centerline[-1].point[2] else centerline


def get_index_for_point(centerline, point):
    for idx, vertex in enumerate(centerline):
        if np.array_equal(vertex.point, point):
            return idx
    return False


def get_mid_of_points(centerline, a, b):
    if b < a:
        a, b = b, a
    target = get_len_path(centerline[a:b]) // 2
    length = 0.0
    for idx in range(max(1, a), b):
        length += np.linalg.norm(centerline[idx].point - centerline[idx - 1].point)
        if length >= target:
            return idx


def get_next_point_by_distance(centerline, start, distance, spacing):
    if distance == 0:
        return start
    length = 0.0
    if distance >= 0:
        indexes = range(start + 1, len(centerline) - 1)
        previous_offset = -1
    else:
        indexes = reversed(range(0, start - 1))
        previous_offset = 1
    for idx in indexes:
        length += get_distance(
            centerline[idx].point, centerline[idx + previous_offset].point, spacing
        )
        if length >= abs(distance):
            return idx
    return None


def add_point_to_centerline(image, centerline, point):
    if point is None:
        return image, centerline
    point = np.asarray(point)
    last = centerline[-1].point
    distance = np.linalg.norm(last - point)
    if distance > 1:
        count = int(distance)
        for idx in range(1, count + 1):
            intermediate = last + idx / float(count + 1) * (point - last)
            image[tuple(intermediate.astype(int))] = 1
            centerline.append(Vertex(intermediate))
    image[tuple(point.astype(int))] = 1
    centerline.append(Vertex(point))
    return image, centerline


def extend_centerline(image, centerline, distance):
    if len(centerline) < 2:
        raise ValueError("Not enough vertices to determine direction")
    result = [Vertex(vertex.point) for vertex in centerline]
    result_image = image.copy()
    direction = result[-1].point - result[-2].point
    direction = direction / np.linalg.norm(direction)
    return add_point_to_centerline(
        result_image, result, result[-1].point + distance * direction
    )


def get_closest_point_of_centerline(point, centerline):
    points = np.asarray([vertex.point for vertex in centerline])
    return tuple(points[np.argmin(np.linalg.norm(points - point, axis=1))])


def get_closest_centerline_section(point, centerline, length_mm, spacing):
    closest = get_index_for_point(
        centerline, get_closest_point_of_centerline(point, centerline)
    )
    count = int(length_mm / spacing)
    start = max(closest - count // 2, 0)
    end = min(closest + count // 2, len(centerline) - 1)
    if end - start < count - 1:
        if start == 0:
            end = min(count - 1, len(centerline) - 1)
        elif end == len(centerline) - 1:
            start = max(end - count + 1, 0)
    return centerline[start : end + 1]


def get_normal_for_cl_point(centerline, idx, dist=(3, 10)):
    normals = []
    for distance in range(dist[0], dist[1]):
        start = max(0, idx - distance)
        end = min(len(centerline) - 1, idx + distance)
        normals.append(centerline[start].point - centerline[end].point)
    return np.asarray(normals).mean(axis=0)


def _distance_to_plane(point, normal, plane_point):
    return ((np.asarray(point) @ normal) - (plane_point @ normal)) / np.linalg.norm(normal)


def get_closest_point_of_centerline_by_plane(point, centerline):
    distances = []
    normals = []
    for idx, vertex in enumerate(centerline[:-1]):
        normal = get_normal_for_cl_point(centerline, idx)
        distances.append(abs(_distance_to_plane(point, normal, vertex.point)))
        normals.append(normal)
    idx = int(np.argmin(distances))
    return tuple(centerline[idx].point), normals[idx]


def resample_cl_to_same_distance(centerline, vox_to_mm_affine, dist=20):
    points = np.asarray([vertex.point for vertex in centerline], dtype=float)
    if len(points) < 2:
        return [Vertex(point) for point in points]
    points_mm = nib.affines.apply_affine(vox_to_mm_affine, points)
    lengths = np.linalg.norm(np.diff(points_mm, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if cumulative[-1] == 0:
        return [Vertex(points[0]), Vertex(points[0])]
    positions = np.linspace(0, cumulative[-1], max(2, int(cumulative[-1] / dist)))
    sampled_mm = np.column_stack(
        [np.interp(positions, cumulative, points_mm[:, dim]) for dim in range(3)]
    )
    sampled = nib.affines.apply_affine(np.linalg.inv(vox_to_mm_affine), sampled_mm)
    return [Vertex(point) for point in sampled]


def check_cl_distances(centerline):
    print(
        np.asarray(
            [
                np.linalg.norm(centerline[idx].point - centerline[idx - 1].point)
                for idx in range(1, len(centerline))
            ]
        )
    )
