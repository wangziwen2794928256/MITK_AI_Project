import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os


# TotalSegmentator输出的肝脏mask

liver_mask = (
r"D:\MITKforTEST\AI-Segmentation\output"
r"\liver.nii.gz"
)


# 分析结果保存

save_dir = (
r"D:\MITK_AI_Project\Results"
r"\Liver_Analysis"
)


os.makedirs(
    save_dir,
    exist_ok=True
)



# ============================
# 读取nii
# ============================


img = nib.load(liver_mask)


data = img.get_fdata()



# voxel尺寸

voxel_size = img.header.get_zooms()


voxel_volume = (
voxel_size[0]
*
voxel_size[1]
*
voxel_size[2]
)



# ============================
# 1. 计算总体积
# ============================


voxel_count = np.sum(data>0)


volume_mm3 = (
voxel_count *
voxel_volume
)


volume_ml = volume_mm3/1000



print("====================")
print("Liver segmentation")
print("====================")

print(
"Voxel number:",
voxel_count
)


print(
"Liver volume:",
round(volume_ml,2),
"ml"
)



# ============================
# 2. 每层面积变化
# ============================


slice_area=[]


for i in range(data.shape[2]):

    pixels=np.sum(
        data[:,:,i]>0
    )

    area = (
    pixels*
    voxel_size[0]*
    voxel_size[1]
    )

    slice_area.append(area/100)



# ============================
# 曲线1
# 切片面积变化
# ============================


plt.figure(figsize=(8,4))


plt.plot(
range(len(slice_area)),
slice_area
)


plt.xlabel(
"Slice index"
)


plt.ylabel(
"Liver area(cm2)"
)


plt.title(
"Liver cross-sectional area variation"
)


plt.grid()


plt.savefig(
os.path.join(
save_dir,
"liver_slice_curve.png"
)
)


plt.close()



# ============================
# 曲线2
# 不同区域体积贡献
# ============================


z=np.arange(
len(slice_area)
)


volume_curve=np.cumsum(
np.array(slice_area)
*
voxel_size[2]
)



plt.figure(figsize=(8,4))


plt.plot(
z,
volume_curve
)


plt.xlabel(
"Slice index"
)


plt.ylabel(
"Accumulated volume(ml)"
)


plt.title(
"Liver volume accumulation curve"
)


plt.grid()


plt.savefig(
os.path.join(
save_dir,
"liver_volume.png"
)
)


plt.close()



print(
"Analysis finished!"
)
