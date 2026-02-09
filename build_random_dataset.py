import os
import numpy as np

# ============================================================================
# Configuration
# ============================================================================
NUM_RANDOM = 18  # number of random samples per dataset

# ============================================================================
# Dataset 1: RGB Images (384x288, float16, ImageNet normalization)
# ============================================================================
rgb_input_path_list = []
RGB_BASE_PATH = "shared_with_container/calibration_data/RandomRGBCalibration"
os.makedirs(RGB_BASE_PATH, exist_ok=True)

IMAGENET_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float16)
IMAGENET_SCALE = np.array([58.395, 57.12, 57.375], dtype=np.float16)

RGB_MIN = IMAGENET_MEAN - IMAGENET_SCALE
RGB_MAX = IMAGENET_MEAN + IMAGENET_SCALE

print("Generating RGB calibration dataset...")

def write_rgb(name, value):
    tensor = np.ones((1, 384, 288, 3), dtype=np.float16) * value
    path = os.path.join(RGB_BASE_PATH, name)
    tensor.tofile(path)
    rgb_input_path_list.append(path)

# Min / Max
write_rgb("rgb_input_min.raw", RGB_MIN)
write_rgb("rgb_input_max.raw", RGB_MAX)

# Random RGB
for i in range(NUM_RANDOM):
    rand_val = np.random.uniform(RGB_MIN, RGB_MAX).astype(np.float16)
    write_rgb(f"rgb_input_rand_{i:03d}.raw", rand_val)

with open("shared_with_container/calibration_data/rgb_input_list.txt", "w") as f:
    for p in rgb_input_path_list:
        f.write(p + "\n")

# ============================================================================
# Dataset 2: Depth Images (384x288, uint16)
# ============================================================================
depth_input_path_list = []
DEPTH_BASE_PATH = "shared_with_container/calibration_data/RandomDepthCalibration"
os.makedirs(DEPTH_BASE_PATH, exist_ok=True)

DEPTH_MEAN = 2500
DEPTH_SCALE = 2000

DEPTH_MIN = DEPTH_MEAN - DEPTH_SCALE
DEPTH_MAX = DEPTH_MEAN + DEPTH_SCALE

print("Generating Depth calibration dataset...")

def write_depth(name, value):
    tensor = np.full((1, 384, 288, 1), value, dtype=np.float16)
    path = os.path.join(DEPTH_BASE_PATH, name)
    tensor.tofile(path)
    depth_input_path_list.append(path)

# Min / Max
write_depth("depth_input_min.raw", DEPTH_MIN)
write_depth("depth_input_max.raw", DEPTH_MAX)

# Random Depth
for i in range(NUM_RANDOM):
    rand_val = np.random.randint(DEPTH_MIN, DEPTH_MAX + 1)
    write_depth(f"depth_input_rand_{i:03d}.raw", rand_val)

with open("shared_with_container/calibration_data/depth_input_list.txt", "w") as f:
    for p in depth_input_path_list:
        f.write(p + "\n")

# ============================================================================
# Dataset 3: Inverse Intrinsics (K_inv, float32)
# ============================================================================
k_inv_input_path_list = []
KINV_BASE_PATH = "shared_with_container/calibration_data/RandomKInvCalibration"
os.makedirs(KINV_BASE_PATH, exist_ok=True)

print("Generating K_inv calibration dataset...")

# ---- Analytic bounds ----
FX_INV_MIN, FX_INV_MAX = 0.000158, 0.00387
FY_INV_MIN, FY_INV_MAX = 0.000211, 0.00387

CX_OVER_FX_MIN, CX_OVER_FX_MAX = -16.1, 15.1
CY_OVER_FY_MIN, CY_OVER_FY_MAX = -9.7, 7.7

def write_k_inv(name, fx_inv, fy_inv, cx_fx, cy_fy):
    k_inv = np.array([
        [fx_inv, cx_fx],
        [fy_inv, cy_fy],
    ], dtype=np.float16)

    path = os.path.join(KINV_BASE_PATH, name)
    k_inv.tofile(path)
    k_inv_input_path_list.append(path)

# Min / Max
write_k_inv("k_inv_min.raw",
            FX_INV_MIN, FY_INV_MIN,
            CX_OVER_FX_MIN, CY_OVER_FY_MIN)

write_k_inv("k_inv_max.raw",
            FX_INV_MAX, FY_INV_MAX,
            CX_OVER_FX_MAX, CY_OVER_FY_MAX)

# Random K_inv
for i in range(NUM_RANDOM):
    fx_inv = np.random.uniform(FX_INV_MIN, FX_INV_MAX)
    fy_inv = np.random.uniform(FY_INV_MIN, FY_INV_MAX)
    cx_fx = np.random.uniform(CX_OVER_FX_MIN, CX_OVER_FX_MAX)
    cy_fy = np.random.uniform(CY_OVER_FY_MIN, CY_OVER_FY_MAX)

    write_k_inv(
        f"k_inv_rand_{i:03d}.raw",
        fx_inv, fy_inv, cx_fx, cy_fy
    )

with open("shared_with_container/calibration_data/k_inv_input_list.txt", "w") as f:
    for p in k_inv_input_path_list:
        f.write(p + "\n")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 60)
print("Summary:")
print(f"  RGB images: {2 + NUM_RANDOM} files (min/max + random)")
print(f"  Depth images: {2 + NUM_RANDOM} files (min/max + random)")
print(f"  K_inv matrices: {2 + NUM_RANDOM} files (min/max + random)")
print("=" * 60)
