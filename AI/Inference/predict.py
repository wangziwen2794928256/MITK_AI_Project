import os
import time
import subprocess
# =====================================================
# 1. 输入输出路径配置
# =====================================================
INPUT_IMAGE = (
    r"D:\MITK_AI_Project\Dataset\Test"
    r"\ct_15mm_defaced.nii"
)
OUTPUT_DIR = (
    r"D:\MITK_AI_Project\Results"
    r"\Prediction_Result"
)
# 创建输出目录
os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)
# =====================================================
# 2. nnUNet环境配置
# =====================================================
def setup_environment():
    """
    设置AI推理运行环境
    TotalSegmentator底层依赖：
    nnUNet v2
    PyTorch
    CUDA
    这里指定模型和计算资源
    """
    env = os.environ.copy()
    # nnUNet模型路径
    env["nnUNet_results"] = (
        r"D:\MITK_AI_Project"
        r"\AI\Models\TotalSegmentator"
    )
    # 防止Windows多进程异常
    env["nnUNet_n_proc_DA"] = "1"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return env
# =====================================================
# 3. 构建TotalSegmentator推理命令
# =====================================================
def build_command():
    cmd = [
        "TotalSegmentator",
        # 输入CT
        "-i",
        INPUT_IMAGE,
        # 输出路径
        "-o",
        OUTPUT_DIR,
        # 使用GPU
        "--device",
        "gpu",
        # 快速模式
        "--fast",
        # 限制线程
        "--nr_thr_resamp",
        "1",
        "--nr_thr_saving",
        "1"
    ]
    return cmd
# =====================================================
# 4. 执行AI推理
# =====================================================
def run_segmentation():
    print("======================")
    print("AI segmentation start")
    print("======================")
    env = setup_environment()
    cmd = build_command()
    start_time = time.time()
    result = subprocess.run(
        cmd,
        env=env
    )
    end_time = time.time()
    inference_time = (
        end_time-start_time
    )
    print("======================")
    print(
        "Inference time:",
        round(inference_time,2),
        "s"
    )
    print(
        "Return code:",
        result.returncode
    )
    if result.returncode == 0:
        print(
            "Segmentation finished!"
        )
    else:
        print(
            "Segmentation failed!"
        )
# =====================================================
# 5. 主函数入口
# =====================================================
if __name__ == "__main__":

    run_segmentation()