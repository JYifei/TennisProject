import numpy as np
import cv2

def project_points(H, points):
    """ 使用透视变换矩阵 H 对一组点进行投影 """
    points = np.array(points, dtype=np.float32)
    ones = np.ones((points.shape[0], 1))
    points_homogeneous = np.hstack([points, ones])  # 转换为齐次坐标
    
    projected_points = H @ points_homogeneous.T  # 透视变换
    projected_points /= projected_points[2, :]  # 归一化
    
    return projected_points[:2, :].T  # 返回 (x, y) 坐标


def adjust_perspective_matrix(H, width_rate, height_rate):
    """ Modify the perspective transform matrix to match the original video resolution. """
    
    # Scale matrix to go from original -> 1280x720
    S_to1280x720 = np.array([
        [1 / width_rate, 0, 0],
        [0, 1 / height_rate, 0],
        [0, 0, 1]
    ])
    
    # Scale matrix to go from 1280x720 -> original
    S_to_original = np.array([
        [width_rate, 0, 0],
        [0, height_rate, 0],
        [0, 0, 1]
    ])
    
    # Compute modified matrix: S_to_original * H * S_to1280x720
    H_modified = S_to_original @ H @ S_to1280x720
    
    return H_modified


# 1280x720 透视矩阵
H_1280x720 = np.array([
    [2.5, 0.3, -100],
    [0.2, 3.1, -200],
    [0.0001, 0.0005, 1]
])

# 原始视频分辨率
width_original = 1920
height_original = 1080
width_rate = width_original / 1280
height_rate = height_original / 720

# 计算修改后的透视矩阵
H_modified = adjust_perspective_matrix(H_1280x720, width_rate, height_rate)

# 选择测试点
test_points = [(0, 0), (1280, 0), (1280, 720), (0, 720), (640, 360)]

# 在 1280x720 下投影
projected_1280 = project_points(H_1280x720, test_points)

# 在修改后的透视矩阵（原始分辨率）下投影
projected_modified = project_points(H_modified, [(x * width_rate, y * height_rate) for x, y in test_points])

# 打印结果
print("Projected points at 1280x720 resolution:\n", projected_1280)
print("Projected points at original resolution:\n", projected_modified)
