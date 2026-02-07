import cv2
import numpy as np
import os
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# 复用你之前的「角点排序」和「透视变换」函数（完全未修改，保持一致性）
def order_corners(corners):
    """对四角点排序：左上→右上→右下→左下"""
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    ordered[0] = corners[np.argmin(s)]  # 左上
    ordered[2] = corners[np.argmax(s)]  # 右下
    diff = np.diff(corners, axis=1)
    ordered[1] = corners[np.argmin(diff)]  # 右上
    ordered[3] = corners[np.argmax(diff)]  # 左下
    return ordered

def warp_document(img, corners):
    """透视矫正纸张为正矩形"""
    (tl, tr, br, bl) = corners
    width1 = np.linalg.norm(br - bl)
    width2 = np.linalg.norm(tr - tl)
    max_width = max(int(width1), int(width2))
    height1 = np.linalg.norm(tr - br)
    height2 = np.linalg.norm(tl - bl)
    max_height = max(int(height1), int(height2))
    dst_corners = np.array([[0, 0], [max_width-1, 0], [max_width-1, max_height-1], [0, max_height-1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners, dst_corners)
    warped = cv2.warpPerspective(img, M, (max_width, max_height))
    return warped

def sam_get_document_corners(img_path, sam_model_path, model_type="vit_b", show_process=True, save_all=True):
    """
    用SAM提取纸张/文档的4个轮廓角点
    :param img_path: 图片绝对路径
    :param sam_model_path: SAM模型权重路径（.pth）
    :param model_type: SAM模型类型（vit_b/vit_l/vit_h）
    :param show_process: 是否显示可视化过程
    :param save_all: 是否保存结果到opencv-output
    :return: 原始图、原始尺寸四角点、矫正后的纸张图
    """
    # 1. 读取图片，保存原始尺寸（用于还原坐标）
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"未找到图片：{img_path}")
    img_original = img.copy()
    h_ori, w_ori = img_original.shape[:2]
    # 缩放图片（统一尺寸，提升SAM速度，不影响最终结果）
    scale = 800 / max(h_ori, w_ori)
    img_resize = cv2.resize(img, (int(w_ori*scale), int(h_ori*scale)))
    img_draw = img_resize.copy()  # 用于绘制掩码、轮廓、角点
    img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)  # SAM要求RGB输入

    # 2. 解析保存路径（完全复用你之前的逻辑，保存到data/opencv-output）
    img_dir = os.path.dirname(img_path)
    output_dir = os.path.join(img_dir, "opencv-output")
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    # 定义保存路径（带sam标识，和其他版本不冲突）
    save_path_mask = os.path.join(output_dir, f"{img_name}_sam2_mask.png")       # SAM掩码图
    save_path_contour = os.path.join(output_dir, f"{img_name}_sam2_contour.png") # 轮廓+角点图
    save_path_warped = os.path.join(output_dir, f"{img_name}_sam2_warped.png")   # 矫正图

    # 3. 加载SAM模型（核心：自动掩码生成，无需手动prompt，零样本）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_model_path)
    sam.to(device=device)
    # 自动掩码生成器（参数调优：适合纸张/文档分割）
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,        # 每侧采样点数，越大分割越细
        pred_iou_thresh=0.9,       # IOU阈值，过滤低质量掩码
        stability_score_thresh=0.9,# 稳定性阈值，过滤不稳定掩码
        crop_n_layers=1,           # 裁剪层数，提升小目标分割
        crop_n_points_downscale_factor=2,
        min_mask_region_area=1000  # 最小掩码面积，过滤小噪点
    )

    # 4. SAM核心：生成所有掩码（自动检测，无需手动点/框prompt）
    masks = mask_generator.generate(img_rgb)
    if not masks:
        raise Exception("SAM未生成任何掩码！请检查图片/模型路径/参数")
    # 筛选掩码：取面积最大的掩码（纸张必然是画面中最大的目标）
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    best_mask = masks[0]['segmentation']  # 最优掩码（布尔矩阵）

    # 5. 从掩码提取轮廓（核心步骤：掩码→轮廓→角点）
    # 将布尔掩码转为二值图（0/255）
    mask_binary = (best_mask.astype(np.uint8)) * 255
    # 提取轮廓（只取外轮廓，压缩点）
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("从SAM掩码未提取到轮廓！")
    # 取最大轮廓（确保是纸张的轮廓）
    max_contour = max(contours, key=cv2.contourArea)

    # 6. 多边形逼近，得到4个角点（纸张是四边形，核心目标）
    perimeter = cv2.arcLength(max_contour, True)
    # 自适应epsilon比例，稳定拟合4个顶点（0.01~0.05，纸张轮廓的合理范围）
    doc_corners = None
    for epsilon_ratio in [0.02, 0.015, 0.025, 0.01, 0.03, 0.04]:
        approx = cv2.approxPolyDP(max_contour, epsilon_ratio * perimeter, True)
        if len(approx) == 4:
            doc_corners = approx.reshape(4, 2)
            break
    if doc_corners is None:
        raise Exception("SAM掩码轮廓无法拟合出4个顶点！请检查图片（是否为纸张/是否严重变形）")

    # 7. 可视化绘制：掩码（半透明）+ 轮廓（绿色）+ 角点（红色）
    # 绘制掩码（半透明蓝色，方便查看分割范围）
    mask_color = np.zeros_like(img_draw)
    mask_color[best_mask] = [255, 0, 0]  # 蓝色掩码
    img_draw = cv2.addWeighted(img_draw, 0.7, mask_color, 0.3, 0)
    # 绘制轮廓+角点
    cv2.drawContours(img_draw, [approx], 0, (0, 255, 0), 2)  # 绿色轮廓
    for (x, y) in doc_corners:
        cv2.circle(img_draw, (x, y), 6, (0, 0, 255), -1)     # 红色角点

    # 8. 角点排序+透视矫正
    doc_corners_ordered = order_corners(doc_corners.astype(np.float32))
    warped_img = warp_document(img_resize, doc_corners_ordered)

    # 9. 保存结果图（到opencv-output）
    if save_all:
        cv2.imwrite(save_path_mask, mask_binary)
        cv2.imwrite(save_path_contour, img_draw)
        cv2.imwrite(save_path_warped, warped_img)
        print(f"\nSAM结果图已保存至：{output_dir}")
        print(f"SAM掩码图：{os.path.basename(save_path_mask)}")
        print(f"轮廓+角点图：{os.path.basename(save_path_contour)}")
        print(f"矫正后纸张图：{os.path.basename(save_path_warped)}")

    # 10. 可视化过程
    if show_process:
        cv2.imshow("1-SAM掩码", mask_binary)
        cv2.imshow("2-SAM轮廓+角点", img_draw)
        cv2.imshow("3-SAM矫正后的纸张", warped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 11. 还原角点坐标到「原始图片尺寸」
    doc_corners_ori = doc_corners_ordered / scale
    doc_corners_ori = doc_corners_ori.astype(np.int32)

    return img_original, doc_corners_ori, warped_img

# ------------------- 测试调用（仅需修改这2个路径） -------------------
if __name__ == "__main__":
    # ********** 仅需修改这两个路径 **********
    SAM_MODEL_PATH = r"H:\PythonProject\vsPythonPro\docDiv\sam_weights\sam_vit_b_01ec64.pth"  # 你的SAM模型路径
    IMG_PATH = r"H:\PythonProject\vsPythonPro\docDiv\data\image4.png"                        # 你的纸张图片路径
    # **************************************

    try:
        original_img, doc_corners, warped_img = sam_get_document_corners(
            img_path=IMG_PATH,
            sam_model_path=SAM_MODEL_PATH,
            model_type="vit_b"  # 对应你的模型类型（vit_b/vit_l/vit_h）
        )
        # 打印原始尺寸的4个角点坐标（可直接使用）
        print("\nSAM提取的纸张四角点坐标（x,y）：")
        print("左上：", doc_corners[0])
        print("右上：", doc_corners[1])
        print("右下：", doc_corners[2])
        print("左下：", doc_corners[3])
    except Exception as e:
        print("执行失败：", e)