import cv2
import numpy as np
import os
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

# ========== 新增：角点排序函数 ==========
def order_corners(corners):
    """对四角点排序：左上→右上→右下→左下"""
    ordered = np.zeros((4, 2), dtype=np.float32)
    # 计算每个点x+y的和：和最小=左上，和最大=右下
    s = corners.sum(axis=1)
    ordered[0] = corners[np.argmin(s)]  # 左上
    ordered[2] = corners[np.argmax(s)]  # 右下
    # 计算每个点y-x的差：差最小=右上，差最大=左下
    diff = np.diff(corners, axis=1)
    ordered[1] = corners[np.argmin(diff)]  # 右上
    ordered[3] = corners[np.argmax(diff)]  # 左下
    return ordered

def sam_generate_mask_canny_and_annotate(img_path, sam_model_path, model_type="vit_b", 
                                         canny_low=50, canny_high=150, show_process=True, save_all=True):
    """
    GPU加速版：SAM生成掩码 → Canny边缘检测 → 原图标注（保留掩码图片，强制GPU运行）
    :param img_path: 图片绝对路径
    :param sam_model_path: SAM模型权重路径（.pth）
    :param model_type: SAM模型类型（vit_b/vit_l/vit_h）
    :param canny_low: Canny低阈值
    :param canny_high: Canny高阈值
    :param show_process: 是否显示可视化窗口
    :param save_all: 是否保存所有结果图（含掩码）
    :return: 原始图、掩码图、Canny边缘图、标注后的原图
    """
    # ========== GPU运行核心配置 ==========
    # 1. 强制检测GPU并验证CUDA可用性
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到CUDA GPU！请确认显卡驱动/安装CUDA版本/PyTorch是否支持GPU")
    
    # 2. 设置GPU设备（支持多GPU时指定第0块）
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"✅ 已启用GPU运行：{torch.cuda.get_device_name(device)}")
    print(f"   GPU显存状态：已用 {torch.cuda.memory_allocated()/1024/1024:.1f}MB / 总 {torch.cuda.get_device_properties(device).total_memory/1024/1024:.1f}MB")

    # ========== 图片预处理 ==========
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"未找到图片：{img_path}")
    img_original = img.copy()
    h_ori, w_ori = img_original.shape[:2]
    
    # 缩放图片（平衡速度/精度，GPU内存不足时可调大scale分母）
    #scale = 1000 / max(h_ori, w_ori)  # GPU足够时可放大到1000像素
    scale = 800 / max(h_ori, w_ori)
    img_resize = cv2.resize(img, (int(w_ori*scale), int(h_ori*scale)))
    img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

    # ========== 保存路径（保留掩码图片） ==========
    img_dir = os.path.dirname(img_path)
    output_dir = os.path.join(img_dir, "opencv-output")
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    save_path_mask = os.path.join(output_dir, f"{img_name}_sam_mask.png")  # 保留掩码图片
    save_path_canny = os.path.join(output_dir, f"{img_name}_sam_canny.png")
    save_path_annotated = os.path.join(output_dir, f"{img_name}_sam_annotated.png")

    # ========== 加载SAM模型（强制GPU） ==========
    # 3. 加载模型到指定GPU，清空缓存避免显存溢出
    torch.cuda.empty_cache()
    sam = sam_model_registry[model_type](checkpoint=sam_model_path)
    sam.to(device=device)  # 强制加载到GPU
    sam.eval()  # 推理模式，减少显存占用

    # 4. SAM掩码生成器（GPU优化参数）
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=48,        # GPU足够时可设48，提升分割精度
        pred_iou_thresh=0.9,
        stability_score_thresh=0.9,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=1000,
        output_mode="binary_mask"  # 减少CPU/GPU数据传输
    )

    # ========== 生成掩码（GPU加速） ==========
    masks = mask_generator.generate(img_rgb)
    if not masks:
        raise Exception("SAM未生成任何掩码！请检查图片/模型路径")
    masks = sorted(masks, key=lambda x: x['area'], reverse=True)
    best_mask = masks[0]['segmentation']
    mask_binary = (best_mask.astype(np.uint8)) * 255  # 保留掩码二值图

    # ========== Canny边缘检测 ==========
    mask_blurred = cv2.GaussianBlur(mask_binary, (5, 5), 0)
    canny_edges = cv2.Canny(mask_blurred, canny_low, canny_high)
    kernel = np.ones((3, 3), np.uint8)
    canny_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)

    # ========== 顶点拟合+原图标注 ==========
    contours, _ = cv2.findContours(canny_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise Exception("从Canny边缘图未提取到轮廓！")
    max_contour = max(contours, key=cv2.contourArea)

    # 容错拟合顶点
    perimeter = cv2.arcLength(max_contour, True)
    doc_corners = None
    for epsilon_ratio in [0.02, 0.015, 0.025, 0.01, 0.03, 0.04]:
        approx = cv2.approxPolyDP(max_contour, epsilon_ratio * perimeter, True)
        if len(approx) == 4:
            doc_corners = approx.reshape(4, 2)
            break
    
    if doc_corners is None:
        print("警告：未拟合出4个顶点，使用凸包近似")
        hull = cv2.convexHull(max_contour)
        hull_perimeter = cv2.arcLength(hull, True)
        for epsilon_ratio in [0.02, 0.03, 0.04]:
            approx = cv2.approxPolyDP(hull, epsilon_ratio * hull_perimeter, True)
            if len(approx) == 4:
                doc_corners = approx.reshape(4, 2)
                break
    
    if doc_corners is None:
        print("警告：凸包近似失败，使用轮廓极值点")
        leftmost = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
        rightmost = tuple(max_contour[max_contour[:, :, 0].argmax()][0])
        topmost = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
        bottommost = tuple(max_contour[max_contour[:, :, 1].argmax()][0])
        doc_corners = np.array([leftmost, rightmost, bottommost, topmost], dtype=np.int32)

    # ========== 新增：角点排序 ==========
    # 先转换为float32（排序函数要求），排序后转回int32
    doc_corners = order_corners(doc_corners.astype(np.float32)).astype(np.int32)

    # 还原坐标+标注
    doc_corners_ori = (doc_corners / scale).astype(np.int32)
    img_annotated = img_original.copy()
    cv2.drawContours(img_annotated, [doc_corners_ori], -1, (0, 255, 0), 2)
    for i, (x, y) in enumerate(doc_corners_ori):
        cv2.circle(img_annotated, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(img_annotated, f"({x},{y})", (x-30, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # ========== 保存结果（强制保留掩码图片） ==========
    if save_all:
        cv2.imwrite(save_path_mask, mask_binary)  # 必保存掩码图片
        cv2.imwrite(save_path_canny, canny_edges)
        cv2.imwrite(save_path_annotated, img_annotated)
        print(f"\n📁 结果已保存至：{output_dir}")
        print(f"   - 掩码图片：{os.path.basename(save_path_mask)}")
        print(f"   - Canny边缘图：{os.path.basename(save_path_canny)}")
        print(f"   - 标注原图：{os.path.basename(save_path_annotated)}")

    # ========== 可视化 ==========
    if show_process:
        cv2.imshow("1-SAM掩码（保留）", mask_binary)
        cv2.imshow("2-Canny边缘", canny_edges)
        cv2.imshow("3-标注后的原图", img_annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 清理GPU缓存
    torch.cuda.empty_cache()
    return img_original, mask_binary, canny_edges, img_annotated

# ------------------- 测试调用（GPU版） -------------------
if __name__ == "__main__":
    # 请确认路径正确
    SAM_MODEL_PATH = r"H:\PythonProject\vsPythonPro\docDiv\sam_weights\sam_vit_b_01ec64.pth"
    IMG_PATH = r"H:\PythonProject\vsPythonPro\docDiv\data\image3.png"

    try:
        original_img, mask_img, canny_img, annotated_img = sam_generate_mask_canny_and_annotate(
            img_path=IMG_PATH,
            sam_model_path=SAM_MODEL_PATH,
            model_type="vit_b",
            canny_low=50,
            canny_high=150
        )
        print("\n✅ 全部流程完成！GPU运行正常，掩码图片已保留，角点已按「左上→右上→右下→左下」排序")
    except Exception as e:
        print(f"\n❌ 执行失败：{e}")
        # 异常时强制清理GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()