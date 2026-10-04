import logging
import socket
import time
import json
from datetime import datetime
from pathlib import Path

import imgkit
import nibabel as nib
import numpy as np
from jinja2 import Environment, FileSystemLoader
from PIL import Image


def to_date(date_as_int):
    if date_as_int:
        return datetime.strptime(str(date_as_int), "%Y%m%d").strftime("%d.%m.%Y")
    return ""


def now():
    return datetime.now().strftime("%d.%m.%Y %H:%M:%S")


def rm_underscores(value):
    return value.replace("_", " ")


def generate_html(
    templates_path,
    template_file,
    template_vars,
    output_file,
    width,
    png_logo_hosts=(),
):
    env = Environment(loader=FileSystemLoader(templates_path))
    env.filters["to_date"] = to_date
    env.globals["now"] = now
    env.globals["rm_underscores"] = rm_underscores
    template = env.get_template(template_file)

    template_vars["styles_path"] = f"{templates_path}/styles.css"
    template_vars["usb_logo_path"] = f"{templates_path}/usb_logo_white.svg"
    template_vars["empty_img_path"] = f"{templates_path}/empty_bg.png"
    logo_suffix = "png" if socket.gethostname() in png_logo_hosts else "svg"
    template_vars["logo_path"] = f"{templates_path}/logo_black.{logo_suffix}"
    html_out = template.render(template_vars)

    with open(str(output_file) + ".html", "w") as f:
        f.write(html_out)

    imgkit.from_string(
        html_out,
        output_file,
        options={
            "xvfb": "",
            "format": "png",
            "width": str(width),
            "enable-local-file-access": "",
            "zoom": "1",
            "quality": "75",
            "quiet": "",
        },
    )
    return Image.open(output_file)


def setup_logger(log_file, name=None):
    log_file = Path(log_file)
    if log_file.exists():
        log_file.unlink()

    logger = logging.getLogger(name or f"totalseg_report.{log_file}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M")
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_runtime(start_time, report_name, nodeinfo_path):
    runtime = time.time() - start_time
    runtime_file = Path(nodeinfo_path).parent / "META" / "runtime.json"
    runtime_file.parent.mkdir(parents=True, exist_ok=True)
    runtime_data = json.load(open(runtime_file)) if runtime_file.exists() else {}
    runtime_data[report_name] = round(runtime, 2)
    json.dump(runtime_data, open(runtime_file, "w"), indent=4)


def rgb_array_to_structured(array):
    rgb_dtype = np.dtype([("R", "u1"), ("G", "u1"), ("B", "u1")])
    return array.copy().view(dtype=rgb_dtype).reshape(array.shape[:-1])


def combine_rgb_slices_as_nifti(slice_paths, logger=None, reverse_z=False):
    files = list(slice_paths)
    if reverse_z:
        files.reverse()

    max_width, max_height = 0, 0
    for filename in files:
        width, height = Image.open(filename).size
        max_width = max(max_width, width)
        max_height = max(max_height, height)
    width, height = max_width + 1, max_height + 1
    if logger is not None:
        logger.info(f"width: {width}, height: {height}")

    nii_img = np.zeros((width, height, len(files), 3), dtype=np.uint8)
    for idx, filename in enumerate(files):
        image = Image.open(filename)
        image.load()
        image = np.asarray(image.convert("RGB"))
        image = image.transpose((1, 0, 2))[::-1, ::-1, :]
        image_x, image_y, _ = image.shape
        nii_img[:image_x, -image_y:, idx, :] = image

    nii_struct = rgb_array_to_structured(nii_img)
    print(f"nii_struct.shape: {nii_struct.shape}")
    new_affine = np.eye(4)
    new_affine[0, 0] = -1
    new_affine[1, 1] = -1
    nii_struct = nii_struct[::-1, ::-1, :]
    return nib.Nifti1Image(nii_struct, new_affine)
