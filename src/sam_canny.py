import cv2
import numpy as np
import os
import torch
import time

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry


def sam_generate_mask_canny_and_annotate(
    img_path,
    sam_model_path,
    output_dir,
    model_type="vit_b",
    canny_low=50,
    canny_high=150,
    show_process=False,
    save_all=True,
):
    """
    单张图片处理：SAM生成掩码 → Canny边缘检测 → 原图标注（强制GPU运行）
    :param img_path: 单张图片绝对路径
    :param sam_model_path: SAM模型权重路径（.pth）
    :param output_dir: 结果保存的根目录（统一指定）
    :param model_type: SAM模型类型（vit_b/vit_l/vit_h）
    :param canny_low: Canny低阈值
    :param canny_high: Canny高阈值
    :param show_process: 是否显示可视化窗口（批量处理建议关闭）
    :param save_all: 是否保存所有结果图（含掩码）
    :return: (处理成功标志, 处理耗时秒数)
    """
    start_time = time.time()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("未检测到CUDA GPU！请确认显卡驱动/安装CUDA版本/PyTorch是否支持GPU")

        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

        img = cv2.imread(img_path)
        if img is None:
            print(f"⚠️ 跳过：未找到图片 {img_path}")
            return False, time.time() - start_time

        img_original = img.copy()
        h_ori, w_ori = img_original.shape[:2]

        scale = 800 / max(h_ori, w_ori)
        img_resize = cv2.resize(img, (int(w_ori * scale), int(h_ori * scale)))
        img_rgb = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)

        os.makedirs(output_dir, exist_ok=True)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        save_path_mask = os.path.join(output_dir, f"{img_name}_sam_mask.png")
        save_path_canny = os.path.join(output_dir, f"{img_name}_sam_canny.png")
        save_path_annotated = os.path.join(output_dir, f"{img_name}_sam_annotated.png")

        torch.cuda.empty_cache()
        sam = sam_model_registry[model_type](checkpoint=sam_model_path)
        sam.to(device=device)
        sam.eval()

        mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=32,
            pred_iou_thresh=0.9,
            stability_score_thresh=0.9,
            crop_n_layers=1,
            crop_n_points_downscale_factor=2,
            min_mask_region_area=1000,
            output_mode="binary_mask",
        )

        masks = mask_generator.generate(img_rgb)
        if not masks:
            print(f"⚠️ 跳过：{img_path} SAM未生成任何掩码")
            return False, time.time() - start_time

        masks = sorted(masks, key=lambda x: x["area"], reverse=True)
        best_mask = masks[0]["segmentation"]
        mask_binary = (best_mask.astype(np.uint8)) * 255

        mask_blurred = cv2.GaussianBlur(mask_binary, (5, 5), 0)
        canny_edges = cv2.Canny(mask_blurred, canny_low, canny_high)
        kernel = np.ones((3, 3), np.uint8)
        canny_edges = cv2.morphologyEx(canny_edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            canny_edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            print(f"⚠️ 跳过：{img_path} 从Canny边缘图未提取到轮廓")
            return False, time.time() - start_time

        max_contour = max(contours, key=cv2.contourArea)

        perimeter = cv2.arcLength(max_contour, True)
        doc_corners = None
        for epsilon_ratio in [0.02, 0.015, 0.025, 0.01, 0.03, 0.04]:
            approx = cv2.approxPolyDP(max_contour, epsilon_ratio * perimeter, True)
            if len(approx) == 4:
                doc_corners = approx.reshape(4, 2)
                break

        if doc_corners is None:
            print(f"⚠️ {img_path}：未拟合出4个顶点，使用凸包近似")
            hull = cv2.convexHull(max_contour)
            hull_perimeter = cv2.arcLength(hull, True)
            for epsilon_ratio in [0.02, 0.03, 0.04]:
                approx = cv2.approxPolyDP(hull, epsilon_ratio * hull_perimeter, True)
                if len(approx) == 4:
                    doc_corners = approx.reshape(4, 2)
                    break

        if doc_corners is None:
            print(f"⚠️ {img_path}：凸包近似失败，使用轮廓极值点")
            leftmost = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
            rightmost = tuple(max_contour[max_contour[:, :, 0].argmax()][0])
            topmost = tuple(max_contour[max_contour[:, :, 1].argmin()][0])
            bottommost = tuple(max_contour[max_contour[:, :, 1].argmax()][0])
            doc_corners = np.array(
                [leftmost, rightmost, bottommost, topmost], dtype=np.int32
            )

        doc_corners_ori = (doc_corners / scale).astype(np.int32)
        img_annotated = img_original.copy()
        cv2.drawContours(img_annotated, [doc_corners_ori], -1, (0, 255, 0), 2)
        for i, (x, y) in enumerate(doc_corners_ori):
            cv2.circle(img_annotated, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(
                img_annotated,
                f"({x},{y})",
                (x - 30, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        if save_all:
            cv2.imwrite(save_path_mask, mask_binary)
            cv2.imwrite(save_path_canny, canny_edges)
            cv2.imwrite(save_path_annotated, img_annotated)
            elapsed_time = time.time() - start_time
            print(f"✅ 已处理：{img_name} → 结果保存至 {output_dir} | 耗时：{elapsed_time:.2f} 秒")

        if show_process:
            cv2.imshow("1-SAM掩码（保留）", mask_binary)
            cv2.imshow("2-Canny边缘", canny_edges)
            cv2.imshow("3-标注后的原图", img_annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        torch.cuda.empty_cache()
        elapsed_time = time.time() - start_time
        return True, elapsed_time

    except Exception as e:
        elapsed_time = time.time() - start_time
        print(f"❌ 处理失败：{os.path.basename(img_path)} → {str(e)} | 耗时：{elapsed_time:.2f} 秒")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return False, elapsed_time


def batch_process_images(
    input_dir,
    output_dir,
    sam_model_path,
    model_type="vit_b",
    canny_low=50,
    canny_high=150,
    show_process=False,
    save_all=True,
):
    """
    批量处理指定目录下的所有图片
    :param input_dir: 输入图片目录
    :param output_dir: 输出结果目录
    :param sam_model_path: SAM模型权重路径
    :param model_type: SAM模型类型
    """
    batch_start_time = time.time()

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录不存在：{input_dir}")

    supported_formats = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
    img_files = [
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.lower().endswith(supported_formats)
    ]

    if not img_files:
        print(f"⚠️ 输入目录 {input_dir} 下未找到支持的图片文件")
        return

    total = len(img_files)
    success_count = 0
    total_elapsed_time = 0.0
    print(f"\n📌 开始批量处理：共 {total} 张图片")
    print(f"📌 输入目录：{input_dir}")
    print(f"📌 输出目录：{output_dir}\n")

    for idx, img_path in enumerate(img_files, 1):
        print(f"[{idx}/{total}] 正在处理：{os.path.basename(img_path)}")
        success, elapsed = sam_generate_mask_canny_and_annotate(
            img_path=img_path,
            sam_model_path=sam_model_path,
            output_dir=output_dir,
            model_type=model_type,
            canny_low=canny_low,
            canny_high=canny_high,
            show_process=show_process,
            save_all=save_all,
        )
        if success:
            success_count += 1
        total_elapsed_time += elapsed

    batch_total_time = time.time() - batch_start_time
    avg_time_per_img = total_elapsed_time / total if total > 0 else 0

    print(f"\n📊 批量处理完成！")
    print(f"✅ 成功：{success_count} 张")
    print(f"❌ 失败：{total - success_count} 张")
    print(f"⏱️  单张平均耗时：{avg_time_per_img:.2f} 秒/张")
    print(f"⏱️  批量总耗时：{batch_total_time:.2f} 秒")
    print(f"📁 所有结果已保存至：{output_dir}")
