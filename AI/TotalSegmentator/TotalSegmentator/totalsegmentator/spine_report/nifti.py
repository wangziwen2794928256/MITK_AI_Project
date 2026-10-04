from totalsegmentator.reporting import combine_rgb_slices_as_nifti, rgb_array_to_structured


def combine_as_nifti(tmp_dir, logger, ref_img):
    return combine_rgb_slices_as_nifti(
        [tmp_dir / "spine_report_frontpage.png"],
        logger=logger,
        reverse_z=False,
    )
