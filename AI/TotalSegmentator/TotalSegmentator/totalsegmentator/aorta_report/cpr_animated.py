import gc
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from skimage import measure
from scipy.ndimage import gaussian_filter1d

from totalsegmentator.aorta_report.geometry import get_region_diameters_pd
from totalsegmentator.aorta_report.plotting import (
    _fit_curve_positions_to_display,
    _get_landmark_positions,
    _plot_mask_contours,
    _safe_smooth_contour,
)
from totalsegmentator.aorta_report.nifti import rgb_array_to_structured
from totalsegmentator.aorta_report.report_layout import compose_report_image, get_report_layout_base


def generate_animated_cpr_nifti(
    aorta, cl, affine,
    res_ct_img, res_seg_img,
    true_lumen_cpr_img, false_lumen_cpr_img,
    cpr_info,
    curve_positions_mm, aorta_curve_mm, true_curve_mm, false_curve_mm,
    landmarks,
    max_dia_all,
    output_path, tmp_dir,
    smoothing=100,
    vmin=-700, vmax=1000,
    logger=None,
    metadata=None,
):
    """Generate an animated CPR as a scrollable nifti stack.

    Each slice shows 4 panels side-by-side:
      1. 3D aorta rendering with an orange cross-sectional plane
      2. Coronal CPR with an orange position line
      3. Diameter profile with an orange position line
      4. 2D cross-sectional view with area / diameter overlay
    """
    from fury import window
    import vtk
    from vtk.util import numpy_support as vtk_np
    from tqdm import tqdm
    from totalsegmentator.vtk_utils import plot_mask as vtk_plot_mask

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cpr_nr_frames = cpr_info["nr_slices"]
    sp_mm = cpr_info["sample_positions_mm"]
    total_len = cpr_info["total_length_mm"]
    step_mm = cpr_info["centerline_step_mm"]

    TARGET_H = 800
    TITLE_H = 28

    # ── Recompute the resampled centerline that the CPR used ─────────
    cl_pts = np.array([v.point for v in cl], dtype=float)
    cl_mm_arr = nib.affines.apply_affine(affine, cl_pts)
    seg_lens = np.linalg.norm(np.diff(cl_mm_arr, axis=0), axis=1)
    cum_mm = np.concatenate(([0.0], np.cumsum(seg_lens)))

    rsp_mm = np.column_stack([
        np.interp(sp_mm, cum_mm, cl_mm_arr[:, d]) for d in range(3)
    ])

    # Smooth centerline in mm-space (same as cpr.py) to avoid plane jitter
    if len(rsp_mm) >= 5:
        rsp_mm = gaussian_filter1d(rsp_mm, sigma=2.0, axis=0, mode="nearest")
        rsp_mm[0] = nib.affines.apply_affine(affine, cl_pts[0])
        rsp_mm[-1] = nib.affines.apply_affine(affine, cl_pts[-1])

    rsp_vox = nib.affines.apply_affine(np.linalg.inv(affine), rsp_mm)

    tan = np.zeros_like(rsp_vox)
    if cpr_nr_frames >= 2:
        tan[0] = rsp_vox[1] - rsp_vox[0]
        tan[-1] = rsp_vox[-1] - rsp_vox[-2]
    if cpr_nr_frames > 2:
        tan[1:-1] = rsp_vox[2:] - rsp_vox[:-2]
    tn = np.linalg.norm(tan, axis=1, keepdims=True)
    tn[tn < 1e-8] = 1.0
    tan /= tn

    # ── VTK coordinate helpers ───────────────────────────────────────
    Sx, Sy, _Sz = aorta.shape

    def _v2vtk(pt):
        return np.array([Sy - 1 - pt[1], pt[2], Sx - 1 - pt[0]], dtype=float)

    def _d2vtk(d):
        return np.array([-d[1], d[2], -d[0]], dtype=float)

    def _orient_transform(center_vtk, normal_vtk):
        """Build a vtkTransform that moves to *center* and rotates z-axis onto *normal*."""
        nv = np.asarray(normal_vtk, dtype=float)
        nl = np.linalg.norm(nv)
        nv = nv / nl if nl > 1e-10 else np.array([0.0, 0.0, 1.0])

        tfm = vtk.vtkTransform()
        tfm.Translate(*center_vtk)
        z0 = np.array([0.0, 0.0, 1.0])
        cr = np.cross(z0, nv)
        crl = np.linalg.norm(cr)
        dot = np.dot(z0, nv)
        if crl > 1e-8:
            tfm.RotateWXYZ(
                np.degrees(np.arccos(np.clip(dot, -1.0, 1.0))),
                *(cr / crl),
            )
        elif dot < 0:
            tfm.RotateX(180)
        return tfm

    def _make_plane(center_vtk, normal_vtk, radius):
        tfm = _orient_transform(center_vtk, normal_vtk)

        # Filled semi-transparent disk
        disk = vtk.vtkDiskSource()
        disk.SetInnerRadius(0)
        disk.SetOuterRadius(radius)
        disk.SetRadialResolution(1)
        disk.SetCircumferentialResolution(64)
        disk.Update()

        tf_disk = vtk.vtkTransformPolyDataFilter()
        tf_disk.SetTransform(tfm)
        tf_disk.SetInputConnection(disk.GetOutputPort())
        tf_disk.Update()

        mapper_disk = vtk.vtkPolyDataMapper()
        mapper_disk.SetInputConnection(tf_disk.GetOutputPort())
        act_disk = vtk.vtkActor()
        act_disk.SetMapper(mapper_disk)
        act_disk.GetProperty().SetColor(1.0, 0.6, 0.0)
        act_disk.GetProperty().SetOpacity(0.85)

        # Bright opaque edge ring for visibility
        ring = vtk.vtkDiskSource()
        ring.SetInnerRadius(radius * 0.88)
        ring.SetOuterRadius(radius)
        ring.SetRadialResolution(1)
        ring.SetCircumferentialResolution(64)
        ring.Update()

        tf_ring = vtk.vtkTransformPolyDataFilter()
        tf_ring.SetTransform(tfm)
        tf_ring.SetInputConnection(ring.GetOutputPort())
        tf_ring.Update()

        mapper_ring = vtk.vtkPolyDataMapper()
        mapper_ring.SetInputConnection(tf_ring.GetOutputPort())
        act_ring = vtk.vtkActor()
        act_ring.SetMapper(mapper_ring)
        act_ring.GetProperty().SetColor(1.0, 0.5, 0.0)
        act_ring.GetProperty().SetOpacity(1.0)

        assembly = vtk.vtkAssembly()
        assembly.AddPart(act_disk)
        assembly.AddPart(act_ring)
        return assembly

    # ── Extract CPR arrays ───────────────────────────────────────────
    ct_cpr = res_ct_img.get_fdata()
    seg_cpr = (res_seg_img.get_fdata() > 0.5).astype(np.uint8)
    in_plane_sp = float(res_ct_img.header.get_zooms()[0])

    img_cor = ct_cpr[:, ct_cpr.shape[1] // 2, :].T[::-1, ::-1]
    mask_cor = seg_cpr[:, seg_cpr.shape[1] // 2, :].T[::-1, ::-1]

    ren_win = None
    scene = None
    try:
        # ── VTK scene setup (offscreen) ──────────────────────────────
        vtk_sz = (350, 700)
        scene = window.Scene()

        aorta_act = vtk_plot_mask(
            scene, aorta, np.eye(4), 0, 0, smoothing=smoothing,
            color=[0.7, 0.7, 0.7], opacity=0.3, orientation="sagittal",
        )
        scene.add(aorta_act)

        plane_radius = max_dia_all * 0.75
        pa_first = _make_plane(_v2vtk(rsp_vox[0]), _d2vtk(tan[0]), plane_radius)
        pa_last = _make_plane(_v2vtk(rsp_vox[-1]), _d2vtk(tan[-1]), plane_radius)
        scene.add(pa_first)
        scene.add(pa_last)

        ren_win = vtk.vtkRenderWindow()
        ren_win.SetOffScreenRendering(1)
        ren_win.AddRenderer(scene)
        ren_win.SetSize(*vtk_sz)
        ren_win.Render()

        scene.projection(proj_type="parallel")
        scene.reset_camera_tight(margin_factor=1.08)
        scene.rm(pa_first)
        scene.rm(pa_last)

        def _snap():
            ren_win.Render()
            w2if = vtk.vtkWindowToImageFilter()
            w2if.SetInput(ren_win)
            w2if.SetInputBufferTypeToRGB()
            w2if.Update()
            vtk_image = w2if.GetOutput()
            dims = vtk_image.GetDimensions()
            sc = vtk_image.GetPointData().GetScalars()
            nc = sc.GetNumberOfComponents()
            arr = vtk_np.vtk_to_numpy(sc).reshape(dims[1], dims[0], nc)
            w2if.SetInput(None)
            return arr[::-1].copy()

        # Compute a fixed crop region from the aorta-only render so all
        # frames use the same bounds (the per-frame plane won't shift it).
        aorta_snap = _snap()
        gray = aorta_snap.max(axis=2)
        rows_nz = np.any(gray > 5, axis=1)
        cols_nz = np.any(gray > 5, axis=0)
        if rows_nz.any() and cols_nz.any():
            crop_r0, crop_r1 = np.where(rows_nz)[0][[0, -1]]
            crop_c0, crop_c1 = np.where(cols_nz)[0][[0, -1]]
            pad_px = 12
            crop_r0 = max(0, crop_r0 - pad_px)
            crop_r1 = min(aorta_snap.shape[0] - 1, crop_r1 + pad_px)
            crop_c0 = max(0, crop_c0 - pad_px)
            crop_c1 = min(aorta_snap.shape[1] - 1, crop_c1 + pad_px)
        else:
            crop_r0, crop_r1 = 0, aorta_snap.shape[0] - 1
            crop_c0, crop_c1 = 0, aorta_snap.shape[1] - 1

        # ── Pre-render CPR coronal base ──────────────────────────────
        cpr_rows, cpr_cols = img_cor.shape
        cpr_scale = TARGET_H / cpr_rows
        cpr_w = max(int(cpr_cols * cpr_scale), 150)

        plt.style.use("dark_background")
        dpi = 100
        fig_c = plt.figure(
            figsize=(cpr_w / dpi, TARGET_H / dpi), dpi=dpi, facecolor="black",
        )
        ax_c = fig_c.add_axes([0, 0, 1, 1])
        ax_c.imshow(
            img_cor, cmap="gray", vmin=vmin, vmax=vmax,
            interpolation="bicubic", aspect="auto",
        )
        _plot_mask_contours(ax_c, mask_cor, color="#ff4d4d", smooth=35, linewidth=1.5)
        ax_c.axis("off")

        fig_c.canvas.draw()
        fig_h_c, fig_w_c = fig_c.canvas.get_width_height()[::-1]
        buf = np.frombuffer(fig_c.canvas.buffer_rgba(), dtype=np.uint8)
        cpr_base = buf.reshape(fig_h_c, fig_w_c, 4)[:, :, :3].copy()

        cpr_py_full = np.zeros(cpr_nr_frames, dtype=int)
        for fi in range(cpr_nr_frames):
            px = ax_c.transData.transform([0, cpr_nr_frames - 1 - fi])
            cpr_py_full[fi] = int(fig_h_c - px[1])
        plt.close(fig_c)

        if cpr_base.shape[0] != TARGET_H:
            s = TARGET_H / cpr_base.shape[0]
            cpr_base = np.array(
                Image.fromarray(cpr_base).resize(
                    (cpr_base.shape[1], TARGET_H), Image.LANCZOS,
                )
            )
            cpr_py_full = np.clip((cpr_py_full * s).astype(int), 0, TARGET_H - 1)

        cpr_display_y_mm = np.arange(cpr_nr_frames)[::-1] * step_mm
        if curve_positions_mm is not None and len(curve_positions_mm) > 0:
            frame_positions_mm = _fit_curve_positions_to_display(
                curve_positions_mm, cpr_display_y_mm,
            )
            frame_indices = np.array([
                int(np.argmin(np.abs(cpr_display_y_mm - frame_pos_mm)))
                for frame_pos_mm in frame_positions_mm
            ], dtype=int)
        else:
            frame_positions_mm = cpr_display_y_mm
            frame_indices = np.arange(cpr_nr_frames, dtype=int)

        nr_frames = len(frame_indices)
        cpr_py = cpr_py_full[frame_indices]
        rsp_vox_frames = rsp_vox[frame_indices]
        tan_frames = tan[frame_indices]

        if logger:
            logger.info(f"Generating animated CPR ({nr_frames} frames) ...")

        # ── Pre-render diameter profile base ─────────────────────────
        # Compute the pixel extent of the CPR data region to align the
        # diameter axes identically.
        data_top_px = int(np.min(cpr_py))
        data_bot_px = int(np.max(cpr_py))
        frac_top = data_top_px / TARGET_H
        frac_bot = data_bot_px / TARGET_H
        ax_bottom = 1.0 - frac_bot
        ax_height = frac_bot - frac_top
        ax_bottom = max(0.01, ax_bottom)
        ax_height = min(ax_height, 1.0 - ax_bottom - 0.01)

        display_y_mm = cpr_display_y_mm
        curve_y_mm = _fit_curve_positions_to_display(curve_positions_mm, display_y_mm)
        display_y_max = float(display_y_mm.max())
        dia_w_px = 400
        fig_d = plt.figure(
            figsize=(dia_w_px / dpi, TARGET_H / dpi), dpi=dpi, facecolor="black",
        )
        ax_d = fig_d.add_axes([0.22, ax_bottom, 0.52, ax_height])

        finite_max = 10.0
        if aorta_curve_mm is not None and np.isfinite(aorta_curve_mm).any():
            ax_d.plot(
                aorta_curve_mm, curve_y_mm,
                color="#f7c66a", linewidth=1.6, alpha=0.85, label="Aorta",
            )
            finite_max = max(
                finite_max,
                float(np.nanmax(aorta_curve_mm[np.isfinite(aorta_curve_mm)])),
            )
        if true_curve_mm is not None and np.isfinite(true_curve_mm).any():
            ax_d.plot(
                true_curve_mm, curve_y_mm,
                color="#7CFC70", linewidth=2.2, label="True lumen",
            )
            finite_max = max(
                finite_max,
                float(np.nanmax(true_curve_mm[np.isfinite(true_curve_mm)])),
            )
        if false_curve_mm is not None and np.isfinite(false_curve_mm).any():
            ax_d.plot(
                false_curve_mm, curve_y_mm,
                color="#ff6b6b", linewidth=2.2, label="False lumen",
            )
            finite_max = max(
                finite_max,
                float(np.nanmax(false_curve_mm[np.isfinite(false_curve_mm)])),
            )

        lm_positions = _get_landmark_positions(
            landmarks, cl_pts, affine, sp_mm, cpr_nr_frames, step_mm,
        )
        for lm in lm_positions:
            ax_d.axhline(lm["distance_mm"], color="white", linewidth=0.6, alpha=0.2)
            ax_d.text(
                finite_max * 1.05, lm["distance_mm"], lm["label"],
                color="white", fontsize=7, va="center", ha="left",
            )

        ax_d.set_xlabel("Diameter [mm]", color="white", fontsize=8)
        ax_d.set_ylabel("Along aorta centerline [mm]", color="white", fontsize=8)
        ax_d.set_xlim(0, finite_max * 1.45)
        ax_d.set_ylim(display_y_max, 0)
        ax_d.tick_params(colors="white", labelsize=7)
        ax_d.grid(axis="x", color="white", alpha=0.1, linewidth=0.5)
        for sp_ in ax_d.spines.values():
            sp_.set_color("white")
            sp_.set_alpha(0.3)
        ax_d.legend(
            frameon=False, loc="upper center",
            bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=7,
        )

        fig_d.canvas.draw()
        fig_h_d, fig_w_d = fig_d.canvas.get_width_height()[::-1]
        buf = np.frombuffer(fig_d.canvas.buffer_rgba(), dtype=np.uint8)
        dia_base = buf.reshape(fig_h_d, fig_w_d, 4)[:, :, :3].copy()
        plt.close(fig_d)

        if dia_base.shape[0] != TARGET_H:
            dia_base = np.array(
                Image.fromarray(dia_base).resize(
                    (dia_base.shape[1], TARGET_H), Image.LANCZOS,
                )
            )

        dia_py = cpr_py.copy()

        W_3D = 200
        W_CPR = cpr_base.shape[1]
        W_DIA = dia_base.shape[1]
        W_CS = 280
        TOTAL_W = W_3D + W_CPR + W_DIA + W_CS

        try:
            title_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13,
            )
            sm_font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12,
            )
        except Exception:
            title_font = ImageFont.load_default()
            sm_font = title_font

        layout_base = get_report_layout_base(
            TOTAL_W,
            TARGET_H + TITLE_H,
            metadata=metadata,
            tmp_dir=tmp_dir,
        )
        all_frames = np.zeros((nr_frames, layout_base[0].height, layout_base[0].width, 3), dtype=np.uint8)

        for fi in tqdm(range(nr_frames), desc="Animated CPR"):
            # ── 3D panel ─────────────────────────────────────────────
            pa = _make_plane(
                _v2vtk(rsp_vox_frames[fi]), _d2vtk(tan_frames[fi]), plane_radius,
            )
            scene.add(pa)
            snap = _snap()
            scene.rm(pa)

            snap = snap[crop_r0 : crop_r1 + 1, crop_c0 : crop_c1 + 1]
            img_3d = np.array(
                Image.fromarray(snap).resize((W_3D, TARGET_H), Image.LANCZOS),
            )

            # ── CPR panel ────────────────────────────────────────────
            cpr_f = cpr_base.copy()
            py = int(np.clip(cpr_py[fi], 1, TARGET_H - 2))
            cpr_f[py - 1 : py + 2, :] = [255, 140, 0]

            # ── Diameter panel ───────────────────────────────────────
            dia_f = dia_base.copy()
            dy = int(np.clip(dia_py[fi], 1, TARGET_H - 2))
            dia_f[dy - 1 : dy + 2, :] = [255, 140, 0]

            # ── Cross-section panel ──────────────────────────────────
            cpr_idx = frame_indices[fi]
            ct_cs = ct_cpr[:, :, cpr_idx]
            seg_cs = seg_cpr[:, :, cpr_idx]
            cs_norm = (
                (ct_cs.astype(float) - vmin) / (vmax - vmin) * 255
            ).clip(0, 255).astype(np.uint8)
            cs_view = cs_norm.T[::-1]
            seg_view = seg_cs.T[::-1]

            cs_disp = min(W_CS - 20, TARGET_H // 2 - 10)
            cs_pil = Image.fromarray(cs_view, mode="L").convert("RGB")
            cs_pil = cs_pil.resize((cs_disp, cs_disp), Image.LANCZOS)

            if seg_view.max() > 0:
                contours_cs = measure.find_contours(seg_view.astype(float), 0.5)
                draw_cs = ImageDraw.Draw(cs_pil)
                sc_factor = cs_disp / seg_view.shape[0]
                for cont in contours_cs:
                    if len(cont) < 3:
                        continue
                    x_s, y_s = _safe_smooth_contour(cont, smooth=20)
                    pts = [
                        (int(xi * sc_factor), int(yi * sc_factor))
                        for xi, yi in zip(x_s, y_s)
                    ]
                    draw_cs.line(pts + [pts[0]], fill=(0, 255, 0), width=2)

            area_cm2 = seg_cs.sum() * in_plane_sp ** 2 / 100
            try:
                dia_val, dia_pd_val, _, _ = get_region_diameters_pd(
                    seg_cs.astype(np.uint8), [in_plane_sp] * 2, np.eye(4), z=0,
                )
            except Exception:
                dia_val, dia_pd_val = 0.0, 0.0

            cs_panel = Image.new("RGB", (W_CS, TARGET_H), (0, 0, 0))
            paste_x = (W_CS - cs_disp) // 2
            cs_panel.paste(cs_pil, (paste_x, 10))
            dp = ImageDraw.Draw(cs_panel)
            ty = 10 + cs_disp + 14
            if area_cm2 > 0.05:
                dp.text(
                    (paste_x, ty),
                    f"\u25A8: {area_cm2:.1f} cm\u00B2",
                    fill=(0, 255, 0), font=sm_font,
                )
                dp.text(
                    (paste_x, ty + 22),
                    f"\u00D8: {dia_val / 10:.1f} x {dia_pd_val / 10:.1f} cm",
                    fill=(0, 255, 0), font=sm_font,
                )
            cs_arr = np.array(cs_panel)

            # ── Compose frame ────────────────────────────────────────
            frame_pil = Image.new(
                "RGB", (TOTAL_W, TARGET_H + TITLE_H), (0, 0, 0),
            )
            draw_f = ImageDraw.Draw(frame_pil)
            x_off = 0
            for title, w in [
                ("3D", W_3D),
                ("Coronal CPR", W_CPR),
                ("Diameter Profile", W_DIA),
                ("Crossectional view", W_CS),
            ]:
                try:
                    tw = draw_f.textlength(title, font=title_font)
                except AttributeError:
                    tw = len(title) * 8
                draw_f.text(
                    (x_off + (w - tw) / 2, 6), title,
                    fill="white", font=title_font,
                )
                x_off += w

            frame = np.array(frame_pil)
            frame[TITLE_H : TITLE_H + TARGET_H, :W_3D] = img_3d
            x2 = W_3D
            frame[TITLE_H : TITLE_H + TARGET_H, x2 : x2 + W_CPR] = (
                cpr_f[:TARGET_H, :W_CPR]
            )
            x2 += W_CPR
            frame[TITLE_H : TITLE_H + TARGET_H, x2 : x2 + W_DIA] = (
                dia_f[:TARGET_H, :W_DIA]
            )
            x2 += W_DIA
            frame[TITLE_H : TITLE_H + TARGET_H, x2 : x2 + W_CS] = (
                cs_arr[:TARGET_H, :W_CS]
            )

            layout_frame = compose_report_image(frame, layout_base=layout_base)
            all_frames[fi] = np.array(layout_frame, dtype=np.uint8)
    finally:
        if scene is not None:
            try:
                scene.clear()
            except Exception:
                pass
        if ren_win is not None:
            try:
                if scene is not None:
                    ren_win.RemoveRenderer(scene)
            except Exception:
                pass
            try:
                ren_win.SetOffScreenRendering(0)
            except Exception:
                pass
            try:
                ren_win.Finalize()
            except Exception:
                pass
        scene = None
        ren_win = None
        gc.collect()

    # ── Convert to nifti (same encoding as combine_as_nifti) ────────
    n_fr, h_fr, w_fr, _ = all_frames.shape
    nii_data = np.zeros((w_fr + 1, h_fr + 1, n_fr, 3), dtype=np.uint8)
    for idx in range(n_fr):
        rev = n_fr - 1 - idx
        img_t = all_frames[rev].transpose(1, 0, 2)[::-1, ::-1, :]
        nii_data[:w_fr, -h_fr:, idx, :] = img_t

    nii_struct = rgb_array_to_structured(nii_data)
    nii_affine = np.eye(4)
    nii_affine[0, 0] = -1
    nii_affine[1, 1] = -1
    nii_struct = nii_struct[::-1, ::-1, :]
    nifti_out = nib.Nifti1Image(nii_struct, nii_affine)
    nib.save(nifti_out, output_path)

    if logger:
        logger.info(f"Animated CPR saved to {output_path}")
