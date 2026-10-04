from totalsegmentator.reporting import (
    combine_rgb_slices_as_nifti,
    rgb_array_to_structured,
)


def combine_as_nifti(slice_images, logger=None):
    return combine_rgb_slices_as_nifti(
        slice_images, logger=logger, reverse_z=True
    )


__all__ = ["combine_as_nifti", "rgb_array_to_structured"]
