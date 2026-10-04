import importlib.resources
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

from totalsegmentator.reporting import generate_html


PLACEHOLDER_RGB = np.array([255, 0, 255], dtype=np.uint8)


def _render_layout_base(content_width, content_height, metadata=None, report_title="Aorta Report", tmp_dir=None):
    assets_dir = importlib.resources.files('totalsegmentator').joinpath('resources/aorta_report')
    page_width = max(int(content_width) + 48, 900)

    metadata = dict(metadata or {})
    metadata.setdefault("version", "")

    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp())
    else:
        tmp_dir = Path(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)

    layout_path = tmp_dir / f"report_layout_{page_width}x{int(content_height)}.png"
    template_vars = {
        "metadata": metadata,
        "report_title": report_title,
        "page_width": page_width,
        "content_width": int(content_width),
        "content_height": int(content_height),
        "placeholder_color": "rgb(255, 0, 255)",
    }
    layout_img = generate_html(
        str(assets_dir),
        "report_template_layout.html",
        template_vars,
        layout_path,
        width=page_width,
        png_logo_hosts=("rigi", "rndapollolp01.uhbs.ch"),
    ).convert("RGB")
    layout_arr = np.array(layout_img)
    placeholder_mask = np.all(np.abs(layout_arr.astype(np.int16) - PLACEHOLDER_RGB.astype(np.int16)) <= 5, axis=2)
    if not placeholder_mask.any():
        raise RuntimeError("Unable to find CPR content placeholder in rendered report layout.")

    ys, xs = np.where(placeholder_mask)
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    return layout_img, bbox


def compose_report_image(content_image, metadata=None, report_title="Aorta Report", tmp_dir=None, layout_base=None):
    if not isinstance(content_image, Image.Image):
        content_image = Image.fromarray(np.asarray(content_image))
    content_image = content_image.convert("RGB")

    if layout_base is None:
        layout_img, bbox = _render_layout_base(
            content_image.width,
            content_image.height,
            metadata=metadata,
            report_title=report_title,
            tmp_dir=tmp_dir,
        )
    else:
        layout_img, bbox = layout_base
        layout_img = layout_img.copy()

    x0, y0, x1, y1 = bbox
    target_size = (x1 - x0, y1 - y0)
    if content_image.size != target_size:
        content_image = content_image.resize(target_size, Image.LANCZOS)
    layout_img.paste(content_image, (x0, y0))
    return layout_img


def get_report_layout_base(content_width, content_height, metadata=None, report_title="Aorta Report", tmp_dir=None):
    return _render_layout_base(
        content_width,
        content_height,
        metadata=metadata,
        report_title=report_title,
        tmp_dir=tmp_dir,
    )
