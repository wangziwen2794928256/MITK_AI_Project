import sys
from pathlib import Path
# p_dir = str(Path(__file__).absolute().parents[1])
# if p_dir not in sys.path: sys.path.insert(0, p_dir)
import time

from PIL import Image
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import measure
from scipy import interpolate

from totalsegmentator.aorta_report.geometry import get_region_diameters_pd
from totalsegmentator.aorta_report.report_layout import compose_report_image


colors = {
    "green": [0, 1, 0],
    "red_light": [1, .27, .18],
    "blue": [0, 0, 1],
    "green_light": [.27, 1, .18],
    "red_dark": [0.6, 0, 0],
    "green_dark": [0, 0.6, 0],
    "blue_light": [.27, .18, 1],
    "brown_light": [0.8, 0.5, 0],
    "brown": [0.3, 0, 0],
    "gray": [0.5, 0.5, 0.5],
    "purple": [0.4, 0, 0.7],
    "blue_dark": [0, 0, 0.6],
    "orange": [0.9, 0.3, 0],
    "yellow": [0.9, 1.0, 0.1],
    "pink": [1, 0.5, 0.7],
    "white": [1, 1, 1],
    "red": [1, 0, 0],
    "gray_light": [0.7, 0.7, 0.7],
    "gray_dark": [0.3, 0.3, 0.3],
}

def smooth_contours(contour, s=20):
    """
    s: 10: no real smoothing
       20: medium smoothing, still close to original
       50: medium smoothing, already a bit too far
      100: stronger smoothing (too for from original)
    """
    x, y = contour[:, 1], contour[:, 0]
    if s == 0:
        return x, y
    tck, u = interpolate.splprep([x, y], s=s)  
    x_new, y_new = interpolate.splev(np.linspace(0, 1, 1000), tck)
    return x_new, y_new
    

def plot_img(img, mask, mask_2, diameter_points, vmin=None, vmax=None, cmap="gray", smooth=20):

    fig = plt.figure()

    # plot img
    plt.imshow(img, cmap=cmap, interpolation="bilinear", origin='lower', vmin=vmin, vmax=vmax)

    # plot contours 1
    contours = measure.find_contours(mask, 0.5)
    if contours is not None:
        for contour in contours:
            x, y = smooth_contours(contour, s=smooth)
            plt.plot(x, y, linewidth=2, color="green")

    # plot contours 2
    if mask_2 is not None:
        contours = measure.find_contours(mask_2, 0.5)
        if contours is not None:
            for contour in contours:
                x, y = smooth_contours(contour, s=smooth)
                plt.plot(x, y, linewidth=2, color="red")

    # plot diameter
    if diameter_points is not None:
        start, end, start_pd, end_pd = diameter_points
        plt.plot([start[1], end[1]], [start[0], end[0]], 'g-') 
        if start_pd is not None:
            plt.plot([start_pd[1], end_pd[1]], [start_pd[0], end_pd[0]], 'g-')

    plt.axis('off')
    return fig


def plot_sagittal_and_coronal_slice(img, mask, output_path, vmin, vmax):
    """
    Plot sagittal and coronal slice of nifti image and nifti binary mask.
    Saves png image with of slices next to each other in one image.

    img: nifti image
    mask: nifti binary mask
    output_path: path to save image
    vmin, vmax: min and max values for image
    """
    image_data = img.get_fdata()
    mask_data = mask.get_fdata()

    img_sagittal = image_data[image_data.shape[0] // 2, :, :].T[::-1, ::-1]
    img_coronal = image_data[:, image_data.shape[1] // 2, :].T[::-1, ::-1]

    mask_sagittal = mask_data[mask_data.shape[0] // 2, :, :].T[::-1, ::-1]
    mask_coronal = mask_data[:, mask_data.shape[1] // 2, :].T[::-1, ::-1]

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(3, 13))

    ax1.imshow(img_sagittal, cmap="gray", vmin=vmin, vmax=vmax)
    ax2.imshow(img_coronal, cmap="gray", vmin=vmin, vmax=vmax)


    # plot mask
    contours = measure.find_contours(mask_sagittal, 0.5)
    if contours is not None:
        for contour in contours:
            ax1.plot(contour[:, 1], contour[:, 0], linewidth=1, color="red")

    contours = measure.find_contours(mask_coronal, 0.5)
    if contours is not None:
        for contour in contours:
            ax2.plot(contour[:, 1], contour[:, 0], linewidth=1, color="red")

    ax1.axis("off")
    ax2.axis("off")

    plt.savefig(output_path, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close()


def _safe_smooth_contour(contour, smooth=20):
    if smooth == 0 or contour.shape[0] < 4:
        return contour[:, 1], contour[:, 0]
    try:
        return smooth_contours(contour, s=smooth)
    except ValueError:
        return contour[:, 1], contour[:, 0]


def _plot_mask_contours(ax, mask_2d, color, smooth=20, linewidth=1.6):
    contours = measure.find_contours(mask_2d, 0.5)
    if contours is None:
        return
    for contour in contours:
        x, y = _safe_smooth_contour(contour, smooth=smooth)
        ax.plot(
            x,
            y,
            linewidth=linewidth,
            color=color,
            # antialiased=True,
            # solid_joinstyle="round",
            # solid_capstyle="round",
        )


def _fit_curve_positions_to_display(curve_positions_mm, display_y_mm):
    """Stretch profile y-positions to the visible CPR extent.

    The diameter profile is sometimes sampled on a slightly shorter centerline
    than the CPR stack, so we map its finite y-range onto the CPR display
    range to keep both panels synchronized at the distal end.
    """
    if curve_positions_mm is None:
        return display_y_mm

    curve_y_mm = np.asarray(curve_positions_mm, dtype=float)
    display_y_mm = np.asarray(display_y_mm, dtype=float)
    if curve_y_mm.size == 0 or display_y_mm.size == 0:
        return curve_y_mm

    finite = np.isfinite(curve_y_mm)
    if finite.sum() < 2:
        return curve_y_mm

    curve_min = float(np.nanmin(curve_y_mm))
    curve_max = float(np.nanmax(curve_y_mm))
    display_min = float(np.nanmin(display_y_mm))
    display_max = float(np.nanmax(display_y_mm))
    if abs(curve_max - curve_min) < 1e-8:
        return np.full_like(curve_y_mm, display_min, dtype=float)

    scale = (display_max - display_min) / (curve_max - curve_min)
    return display_min + (curve_y_mm - curve_min) * scale


def _get_cpr_longitudinal_views(image_data, mask_data):
    img_sagittal = image_data[image_data.shape[0] // 2, :, :].T[::-1, ::-1]
    img_coronal = image_data[:, image_data.shape[1] // 2, :].T[::-1, ::-1]
    mask_sagittal = mask_data[mask_data.shape[0] // 2, :, :].T[::-1, ::-1]
    mask_coronal = mask_data[:, mask_data.shape[1] // 2, :].T[::-1, ::-1]
    return img_sagittal, img_coronal, mask_sagittal, mask_coronal


def _get_cpr_diameter_curve(mask_img):
    mask_data = mask_img.get_fdata() > 0.5
    spacing = np.asarray(mask_img.header.get_zooms()[:2], dtype=float)
    nr_slices = mask_data.shape[2]
    diameters_mm = np.full(nr_slices, np.nan, dtype=float)

    for slice_idx in range(nr_slices):
        mask_slice = mask_data[:, :, slice_idx].astype(np.uint8)
        if mask_slice.sum() < 5:
            continue
        diameter_mm, _, _, _ = get_region_diameters_pd(mask_slice, spacing, np.eye(4), z=0)
        if diameter_mm > 0:
            diameters_mm[slice_idx] = float(diameter_mm)

    return diameters_mm


def _get_landmark_positions(landmarks, centerline_points_vox, centerline_affine, sample_positions_mm, nr_slices, step_mm):
    if landmarks is None or centerline_points_vox is None or centerline_affine is None or sample_positions_mm is None:
        return []

    centerline_points_vox = np.asarray(centerline_points_vox, dtype=float)
    if centerline_points_vox.shape[0] == 0:
        return []

    centerline_points_mm = nib.affines.apply_affine(centerline_affine, centerline_points_vox)
    segment_lengths = np.linalg.norm(np.diff(centerline_points_mm, axis=0), axis=1)
    cumulative_length_mm = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    landmark_positions = []

    for lm_nr, lm_dict in landmarks.items():
        if lm_dict.get("empty") or lm_dict.get("cl_idx") is None:
            continue

        cl_idx = int(np.clip(lm_dict["cl_idx"], 0, len(cumulative_length_mm) - 1))
        landmark_distance_mm = cumulative_length_mm[cl_idx]
        cpr_slice_idx = int(np.argmin(np.abs(sample_positions_mm - landmark_distance_mm)))
        display_slice_idx = nr_slices - 1 - cpr_slice_idx
        label = f"{lm_nr} {lm_dict.get('name', '')}".strip()
        landmark_positions.append({
            "label": label,
            "slice_idx": display_slice_idx,
            "distance_mm": display_slice_idx * step_mm,
        })

    return landmark_positions


def plot_cpr_overview(img, mask, output_path, vmin, vmax, true_lumen_mask=None, false_lumen_mask=None,
                      landmarks=None, centerline_points_vox=None, centerline_affine=None, sample_positions_mm=None,
                      curve_positions_mm=None, aorta_curve_mm=None, true_curve_mm=None, false_curve_mm=None,
                      metadata=None, tmp_dir=None):
    """
    Create a CPR overview PNG with two longitudinal views and a diameter profile.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_data = img.get_fdata()
    mask_data = mask.get_fdata() > 0.5
    img_sagittal, img_coronal, mask_sagittal, mask_coronal = _get_cpr_longitudinal_views(image_data, mask_data)

    nr_slices = image_data.shape[2]
    step_mm = float(img.header.get_zooms()[2])
    display_y_mm = np.arange(nr_slices)[::-1] * step_mm

    aorta_curve = aorta_curve_mm if aorta_curve_mm is not None else _get_cpr_diameter_curve(mask)
    true_curve = true_curve_mm if true_curve_mm is not None else (
        _get_cpr_diameter_curve(true_lumen_mask) if true_lumen_mask is not None else None
    )
    false_curve = false_curve_mm if false_curve_mm is not None else (
        _get_cpr_diameter_curve(false_lumen_mask) if false_lumen_mask is not None else None
    )
    curve_y_mm = _fit_curve_positions_to_display(curve_positions_mm, display_y_mm)
    landmark_positions = _get_landmark_positions(
        landmarks,
        centerline_points_vox,
        centerline_affine,
        sample_positions_mm,
        nr_slices,
        step_mm,
    )

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(13.5, 8.5), facecolor="black")
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.2], wspace=0.16)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    for ax, title, curr_img, curr_mask in (
        (ax1, "Sagittal CPR", img_sagittal, mask_sagittal),
        (ax2, "Coronal CPR", img_coronal, mask_coronal),
    ):
        ax.imshow(curr_img, cmap="gray", vmin=vmin, vmax=vmax, interpolation="bicubic")
        _plot_mask_contours(ax, curr_mask, color="#ff4d4d", smooth=35, linewidth=1.8)
        for landmark in landmark_positions:
            ax.axhline(landmark["slice_idx"], color="white", linewidth=0.8, alpha=0.25)
        ax.set_title(title, color="white", fontsize=12, pad=10)
        ax.axis("off")

    plotted_curves = []
    finite_curves = []

    if np.isfinite(aorta_curve).any():
        plotted_curves.append(ax3.plot(aorta_curve, curve_y_mm, color="#f7c66a", linewidth=1.6, alpha=0.85, label="Aorta")[0])
        finite_curves.append(aorta_curve[np.isfinite(aorta_curve)])
    if true_curve is not None and np.isfinite(true_curve).any():
        plotted_curves.append(ax3.plot(true_curve, curve_y_mm, color="#7CFC70", linewidth=2.2, label="True lumen")[0])
        finite_curves.append(true_curve[np.isfinite(true_curve)])
    if false_curve is not None and np.isfinite(false_curve).any():
        plotted_curves.append(ax3.plot(false_curve, curve_y_mm, color="#ff6b6b", linewidth=2.2, label="False lumen")[0])
        finite_curves.append(false_curve[np.isfinite(false_curve)])

    max_curve_mm = max((float(np.max(curve)) for curve in finite_curves), default=10.0)
    max_curve_mm = max(max_curve_mm, 10.0)

    for landmark in landmark_positions:
        ax3.axhline(landmark["distance_mm"], color="white", linewidth=0.8, alpha=0.25)
        ax3.text(max_curve_mm * 1.05, landmark["distance_mm"], landmark["label"],
                 color="white", fontsize=8, va="center", ha="left")

    ax3.set_title("Diameter Profile", color="white", fontsize=12, pad=10)
    ax3.set_xlabel("Diameter [mm]", color="white")
    ax3.set_ylabel("Along aorta centerline [mm]", color="white")
    ax3.set_xlim(0, max_curve_mm * 1.45)
    ax3.set_ylim(display_y_mm.max(), 0)
    ax3.tick_params(colors="white")
    ax3.grid(axis="x", color="white", alpha=0.15, linewidth=0.6)
    for spine in ax3.spines.values():
        spine.set_color("white")
        spine.set_alpha(0.4)
    if plotted_curves:
        ax3.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=9)

    fig.canvas.draw()
    fig_h, fig_w = fig.canvas.get_width_height()[::-1]
    fig_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(fig_h, fig_w, 4)
    content_img = Image.fromarray(fig_rgba[:, :, :3].copy())
    plt.close(fig)
    final_img = compose_report_image(content_img, metadata=metadata, tmp_dir=tmp_dir)
    final_img.save(output_path)


def plot_masks_3d(masks, output_path, file_prefix, smoothing=20, nr_frames=12, debug=False, colors_subset=None):
    """
    todo: check that all masks are not empty -> otherwise error

    masks: list of binary mask ([ndarray])
    """
    from fury import window, actor, ui, io, utils
    from totalsegmentator.vtk_utils import contour_from_roi_smooth, plot_mask

    window_size = (700, 900)
    scene = window.Scene()
    showm = window.ShowManager(scene=scene, size=window_size, reset_camera=False)
    showm.initialize()
    
    for idx, mask in enumerate(masks):
        color = list(colors.keys())[idx] if colors_subset is None else colors_subset[idx]
        scene.add(plot_mask(scene, mask, np.eye(4), 0, 0, smoothing=smoothing,
                color=colors[color], opacity=1.0, orientation="sagittal"))

    # Save single shot
    scene.reset_camera_tight(margin_factor=1.02)  # need to do reset_camera=False in record for this to work in
    az_ang = int(360 / nr_frames * 1.2)  # increase 20% to make a bit more than one full rotation
    # Video
    output_path = str(output_path / file_prefix)
    window.record(scene=scene, size=window_size, out_path=output_path, reset_camera=True,
                  path_numbering=True, n_frames=nr_frames, az_ang=az_ang)
    for i in range(nr_frames):
        img = np.array(Image.open(output_path + f"{i:06d}.png"))
        img = img[100:-100, 150:-150]  # top, bottom, left, right
        Image.fromarray(img).save(output_path + f"{i:06d}.png")

    scene.clear()
    
    
def plot_aorta_3d(aorta, true_lumen, false_lumen, all_vessels, centerline, landmarks, output_path, smoothing=20, nr_frames=12, debug=False):
    """
    todo: check that all masks are not empty -> otherwise error

    aorta: binary mask (ndarray)
    all_vessels: binary mask (ndarray)
    landsmakrs: dict of all landsmarks. each landmark has keys cl_idx, roi, diameter
    """
    from fury import window, actor, ui, io, utils
    from totalsegmentator.vtk_utils import contour_from_roi_smooth, plot_mask

    for lm_nr, lm_dict in landmarks.items():
        lm_dict["color"] = list(colors.values())[lm_nr-1]

    # Offsets to move text nicely next to ROIs
    # x: left(-)/right(+), y: up(+)/down(-), z: front/back
    # landmarks[1]["txt_offset"] = np.array([-80,-10,0])
    # landmarks[2]["txt_offset"] = np.array([-80,-5,0])
    # landmarks[3]["txt_offset"] = np.array([-80,0,0])
    # landmarks[4]["txt_offset"] = np.array([-80,0,0])
    # landmarks[5]["txt_offset"] = np.array([-80,0,0])
    # landmarks[6]["txt_offset"] = np.array([-40,20,0])
    # landmarks[7]["txt_offset"] = np.array([0,0,0])
    # landmarks[8]["txt_offset"] = np.array([20,0,0])
    # landmarks[9]["txt_offset"] = np.array([20,0,0])
    # landmarks[10]["txt_offset"] = np.array([20,0,0])
    # landmarks[11]["txt_offset"] = np.array([20,0,0])

    # offset if only showing number
    landmarks[1]["txt_offset"] = np.array([-50,-10,0])
    landmarks[2]["txt_offset"] = np.array([-50,-5,0])
    landmarks[3]["txt_offset"] = np.array([-50,0,0])
    landmarks[4]["txt_offset"] = np.array([-50,0,0])
    landmarks[5]["txt_offset"] = np.array([-40,0,0])
    landmarks[6]["txt_offset"] = np.array([-10,20,0])
    landmarks[7]["txt_offset"] = np.array([10,0,0])
    landmarks[8]["txt_offset"] = np.array([20,0,0])
    landmarks[9]["txt_offset"] = np.array([20,0,0])
    landmarks[10]["txt_offset"] = np.array([20,0,0])
    landmarks[11]["txt_offset"] = np.array([20,0,0])

    window_size = (700, 900)
    scene = window.Scene()
    showm = window.ShowManager(scene=scene, size=window_size, reset_camera=False)
    showm.initialize()
    
    scene.add(plot_mask(scene, aorta, np.eye(4), 0, 0, smoothing=smoothing,
              color=colors["gray_light"], opacity=.3, orientation="sagittal"))
    
    # scene.add(plot_mask(scene, true_lumen, np.eye(4), 0, 0, smoothing=smoothing,
    #           color=colors["green"], opacity=.3, orientation="sagittal"))
    
    scene.add(plot_mask(scene, false_lumen, np.eye(4), 0, 0, smoothing=smoothing,
              color=colors["red"], opacity=.1, orientation="sagittal"))

    scene.add(plot_mask(scene, all_vessels, np.eye(4), 0, 0, smoothing=smoothing,
              color=colors["gray_dark"], opacity=1.0, orientation="sagittal"))
    
    scene.add(plot_mask(scene, centerline, np.eye(4), 0, 0, smoothing=0,
              color=colors["red"], opacity=1.0, orientation="sagittal"))

    for lm_nr, lm_dict in landmarks.items():
        if not lm_dict["empty"]:
            cont_actor = plot_mask(scene, lm_dict["roi"], np.eye(4), 0, 0, color=lm_dict["color"],
                                orientation="sagittal", smoothing=smoothing)
            scene.add(cont_actor)

            # Apply the same transformations to these points as we applied to the entire image in plot_mask()
            ss = all_vessels.shape
            x,y,z = lm_dict["cl_point"]
            x = ss[0] - x  # invert x
            x,y,z = y,z,x
            x = ss[1] - x  # invert y
            position = np.array([x,y,z]) + lm_dict["txt_offset"]
            # text_actor = actor.text_3d(f"Pos{lm_nr}: {lm_dict['diameter']:.1f}mm", position,
            #                            colors["white"], 6, shadow=False)  # does not face the camera
            text_actor = actor.vector_text(text=f"{lm_nr}", pos=position,
                                        scale=(8,8,8), color=colors["white"])  # always faces the camera
            scene.add(text_actor)

    # Show rotating video
    fps = 20  # Frames per second
    # counter = itertools.count()
    # def timer_callback(_obj, _event):
    #     cnt = next(counter)
    #     showm.scene.azimuth(1.5)  # Azimuth is a constant factor for how fast to rotate the scene (higher is faster)
    #     showm.render()
    #     # print(showm.scene.get_camera())
    #     if cnt == 5000:
    #         showm.exit()

    # Show a bit rotated for better frontal view on aorta
    # (not tested yet if this is also a good view for all subjects)
    # -> this is nice if only want to record a single image, not a video
    # showm.scene.set_camera(position=(-182., 109., 677.),
    #                        focal_point=(124., 109., 145.),
    #                        view_up=(0.0, 1.0, 0.0))

    # showm.add_timer_callback(True, int(1000 // fps), timer_callback)  # Run every 100ms
    # showm.start()

    # Save single shot
    # scene.projection(proj_type='parallel')
    scene.reset_camera_tight(margin_factor=1.02)  # need to do reset_camera=False in record for this to work in
    # output_path = str(output_path / "preview_3d.png")
    # window.record(scene=scene, size=window_size, out_path=output_path, reset_camera=False)

    # crop left and right size to remove black borders
    # img = np.array(Image.open(output_path))
    # img = img[:, 100:-100]
    # Image.fromarray(img).save(output_path)

    az_ang = int(360 / nr_frames * 1.2)  # increase 20% to make a bit more than one full rotation
    # Video
    output_path = str(output_path / "preview_3d_rotating_")
    window.record(scene=scene, size=window_size, out_path=output_path, reset_camera=True,
                  path_numbering=True, n_frames=nr_frames, az_ang=az_ang)
    for i in range(nr_frames):
        img = np.array(Image.open(output_path + f"{i:06d}.png"))
        img = img[100:-100, 150:-150]  # top, bottom, left, right
        Image.fromarray(img).save(output_path + f"{i:06d}.png")

    scene.clear()
