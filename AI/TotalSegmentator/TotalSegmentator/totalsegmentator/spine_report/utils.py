import numpy as np

from totalsegmentator.reporting import save_runtime


def get_erosion_struct_elem(nifti_img, erosion_mm=2):
    """Create an anisotropic 3D erosion kernel in voxel space."""
    voxel_spacings = nifti_img.header.get_zooms()
    radii = [erosion_mm / spacing for spacing in voxel_spacings]

    x, y, z = np.ogrid[
        -radii[0]:radii[0] + 1,
        -radii[1]:radii[1] + 1,
        -radii[2]:radii[2] + 1,
    ]

    return x * x / (radii[0] ** 2) + y * y / (radii[1] ** 2) + z * z / (radii[2] ** 2) <= 1
