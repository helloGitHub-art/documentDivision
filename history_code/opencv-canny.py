import cv2
import numpy as np
import os  # 导入os模块处理路径

# 图片路径
img_path = r"H:\PythonProject\vsPythonPro\docDiv\data\opencv-output\image3_sam_mask.png"

def detect_paper_corners(img_path):
    """
    检测图片中纸张的四个顶点
    :param img_path: 图片路径
    :return: 四个顶点坐标(按左上、右上、右下、左下排序)，绘制了顶点的图像
    """
    # 1. 读取并预处理图像
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError("无法读取图片，请检查路径是否正确！")
    
    # 备份原图用于绘制结果
    img_copy = img.copy()
    # 转灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊降噪（Canny对噪声敏感，必须降噪）
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 2. Canny边缘检测
    edges = cv2.Canny(blurred, 100, 200)
    
    # 3. 形态学闭运算（膨胀+腐蚀），增强边缘连续性
    kernel = np.ones((5, 5), np.uint8)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # 4. 查找轮廓
    contours, _ = cv2.findContours(closed_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 筛选面积最大的轮廓
    if not contours:
        raise ValueError("未检测到任何轮廓，请检查图片或调整Canny阈值！")
    max_contour = max(contours, key=cv2.contourArea)
    
    # 5. 轮廓逼近，找到四边形顶点
    perimeter = cv2.arcLength(max_contour, True)
    approx = cv2.approxPolyDP(max_contour, 0.05* perimeter, True)
    
    # 验证是否是四边形
    if len(approx) != 4:
        raise ValueError(f"检测到{len(approx)}个顶点，未找到纸张的4个顶点！可调整epsilon值重试")
    
    # 提取顶点坐标并排序
    corners = approx.reshape(4, 2).astype(int)
    corners = sorted(corners, key=lambda x: (x[0] + x[1]))
    top_left, bottom_right = corners[0], corners[-1]
    remaining = corners[1:-1]
    top_right = max(remaining, key=lambda x: x[0])
    bottom_left = min(remaining, key=lambda x: x[0])
    sorted_corners = np.array([top_left, top_right, bottom_right, bottom_left], dtype=int)
    
    # 6. 绘制顶点和轮廓
    cv2.drawContours(img_copy, [max_contour], -1, (0, 255, 0), 2)
    for (x, y) in sorted_corners:
        cv2.circle(img_copy, (x, y), 8, (0, 0, 255), -1)
        cv2.putText(img_copy, f"({x},{y})", (x-20, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return sorted_corners, img_copy, edges  # 额外返回edges图，方便保存

# 执行检测
try:
    corners, result_img, edges_img = detect_paper_corners(img_path)
    print("纸张的四个顶点坐标（左上、右上、右下、左下）：")
    for i, (x, y) in enumerate(corners):
        print(f"顶点{i+1}：({x}, {y})")
    
    # ========== 核心修改：提取原图片名并构造新保存路径 ==========
    # 提取原图片的文件名（如image1.png）
    img_filename = os.path.basename(img_path)
    # 分离文件名和扩展名，得到核心名称（如image1）
    img_name_prefix = os.path.splitext(img_filename)[0]  # 结果为"image1"
    
    # 获取当前脚本所在目录（即执行代码的目录）
    current_dir = os.getcwd()
    
    # 构造各图片的保存路径：当前目录 + image_i + 后缀
    original_save_path = os.path.join(current_dir, f"{img_name_prefix}-original.png")
    canny_save_path = os.path.join(current_dir, f"{img_name_prefix}-canny.png")
    result_save_path = os.path.join(current_dir, f"{img_name_prefix}-result.png")
    
    # 保存图片
    cv2.imwrite(original_save_path, cv2.imread(img_path))  # 原始图
    cv2.imwrite(canny_save_path, edges_img)                # Canny边缘检测图
    cv2.imwrite(result_save_path, result_img)              # 顶点检测结果图
    
    print("\n图片已按指定格式保存到当前目录：")
    print(f"- 原始图：{original_save_path}")
    print(f"- Canny边缘检测图：{canny_save_path}")
    print(f"- 顶点检测结果图：{result_save_path}")
    
except Exception as e:
    print(f"错误：{e}")