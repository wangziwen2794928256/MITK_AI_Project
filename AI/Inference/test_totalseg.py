import subprocess
import os


print("==============================")
print("AI segmentation test")
print("==============================")


# 输入CT
input_file = r"D:\MITKforTEST\AI-Segmentation\input\ct_15mm_defaced.nii"

# 输出
output_dir = r"D:\MITKforTEST\AI-Segmentation\output"

# 环境
env = os.environ.copy()

env["PATH"] = (
    r"C:\ANACONDA\envs\mitk-ai;"
    r"C:\ANACONDA\envs\mitk-ai\Library\bin;"
    r"C:\ANACONDA\envs\mitk-ai\Scripts;"
    + env["PATH"]
)

# Windows + nnUNet关键设置
env["nnUNet_n_proc_DA"] = "1"
env["OMP_NUM_THREADS"] = "1"
env["MKL_NUM_THREADS"] = "1"



cmd = [

    "TotalSegmentator",

    "-i",
    input_file,

    "-o",
    output_dir,

    "--device",
    "gpu",

    "--fast",

    "--nr_thr_resamp",
    "1",

    "--nr_thr_saving",
    "1"
]


print("AI segmentation starting...")


result = subprocess.run(
    cmd,
    env=env
)


print("return code:", result.returncode)


if result.returncode == 0:

    print("AI segmentation finished")

else:

    print("AI segmentation failed")