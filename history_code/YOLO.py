from ultralytics import YOLO
import cv2
import numpy as np
import os

# 复用之前的角点排序函数（保证和OpenCV/EasyOCR版本顺序一致：左上→右上→右下→左下）
def order_corners(corners):
    ordered = np.zeros((4, 2), dtype=np.float32)
    s = corners.sum(axis=1)
    ordered[0] = corners[np.argmin(s)]  # 左上
    ordered[2] = corners[np.argmax(s)]  # 右下
    diff = np.diff(corners, axis=1)
    ordered[1] = corners[np.argmin(diff)]  # 右上
    ordered[3] = corners[np.argmax(diff)]  # 左下
    return ordered

# 复用之前的透视变换函数（矫正纸张）
def warp_document(img, corners):
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

def yolov8_pose_infer(img_path, model_path, show_process=True, save_all=True):
    """
    YOLOv8-Pose推理提取纸张四角点，保存结果到data/opencv-output
    :param img_path: 纸张图片绝对路径
    :param model_path: 训练好的最优模型路径（best.pt）
    :param show_process: 是否显示检测/矫正过程
    :param save_all: 是否保存结果图
    :return: 原始图、原始尺寸四角点坐标、矫正后的纸张图
    """
    # 1. 加载YOLOv8-Pose模型
    model = YOLO(model_path)
    # 2. 读取图片，保存原始尺寸（用于还原关键点坐标）
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"未找到图片：{img_path}")
    img_original = img.copy()
    h_ori, w_ori = img_original.shape[:2]  # 原始图片尺寸
    img_draw = img_original.copy()  # 用于绘制检测结果

    # 3. 解析保存路径：自动保存到data/opencv-output（和之前版本同目录）
    img_dir = os.path.dirname(img_path)
    output_dir = os.path.join(img_dir, "opencv-output")
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.splitext(os.path.basename(img_path))[0]
    # 定义保存路径（带yolov8-pose标识，和其他版本不冲突）
    save_path_detect = os.path.join(output_dir, f"{img_name}_yolov8_detect.png")
    save_path_warped = os.path.join(output_dir, f"{img_name}_yolov8_warped.png")

    # 4. YOLOv8-Pose核心推理：检测纸张+提取四角点
    # stream=True：流式推理，节省内存；imgsz=640：推理尺寸（和训练一致）
    results = model(img, stream=True, imgsz=640, conf=0.000000001, iou=0.5)  # conf=0.5：置信度阈值，过滤低精度检测

    paper_corners_ori = None  # 原始图片尺寸的四角点坐标
    for res in results:
        if res.keypoints is None or len(res.keypoints) == 0:
            raise Exception("YOLOv8-Pose未检测到纸张或四角点！")
        # 提取关键点坐标：res.keypoints.xy → 形状(1,4,2)，对应（纸张数，4个关键点，x,y）
        kpts = res.keypoints.xy[0].cpu().numpy()  # 转为numpy数组，(4,2)
        # 关键点坐标还原到原始图片尺寸（因推理时缩放了imgsz=640）
        kpts[:, 0] = kpts[:, 0] * (w_ori / res.orig_shape[1])
        kpts[:, 1] = kpts[:, 1] * (h_ori / res.orig_shape[0])
        kpts = kpts.astype(np.int32)  # 转为整数坐标，方便使用

        # 角点排序（保证顺序：左上→右上→右下→左下，和标注/训练一致）
        kpts_ordered = order_corners(kpts)
        paper_corners_ori = kpts_ordered.astype(np.int32)

        # 绘制检测结果：绿色框（纸张）+ 红色点（四角点）+ 蓝色数字（关键点顺序）
        # 绘制纸张检测框
        box = res.boxes.xyxy[0].cpu().numpy().astype(np.int32)
        cv2.rectangle(img_draw, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
        # 绘制四角点+顺序数字
        for i, (x, y) in enumerate(paper_corners_ori):
            cv2.circle(img_draw, (x, y), 6, (0, 0, 255), -1)  # 红色实心点
            cv2.putText(img_draw, str(i+1), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    if paper_corners_ori is None:
        raise Exception("未检测到纸张的四个顶点角！")

    # 5. 透视矫正纸张
    warped_img = warp_document(img_original, paper_corners_ori.astype(np.float32))

    # 6. 保存结果图到opencv-output
    if save_all:
        cv2.imwrite(save_path_detect, img_draw)
        cv2.imwrite(save_path_warped, warped_img)
        print(f"\nYOLOv8-Pose结果图已保存至：{output_dir}")
        print(f"纸张检测图：{os.path.basename(save_path_detect)}")
        print(f"纸张矫正图：{os.path.basename(save_path_warped)}")

    # 7. 可视化检测过程
    if show_process:
        cv2.imshow("1-YOLOv8-Pose纸张检测+四角点", img_draw)
        cv2.imshow("2-YOLOv8-Pose矫正后的纸张", warped_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return img_original, paper_corners_ori, warped_img

# ------------------- 测试调用（核心：修改模型路径和图片路径） -------------------
if __name__ == "__main__":
    # ********** 只需修改这两个路径 **********
    MODEL_PATH = r"H:\PythonProject\vsPythonPro\docDiv\paper_dataset\afterLabel\runs\pose\paper_train\yolov8n_pose_paper\weights\best.pt"  # 训练好的best.pt路径
    IMG_PATH = r"H:\PythonProject\vsPythonPro\docDiv\data\opencv-output\image5_sam2_mask.png"  # 你的纸张图片路径
    # **************************************

    try:
        original_img, paper_corners, warped_img = yolov8_pose_infer(IMG_PATH, MODEL_PATH)
        # 打印原始图片尺寸的纸张四角点坐标（可直接使用）
        print("\nYOLOv8-Pose提取的纸张四个顶点角坐标（x,y）：")
        print("左上：", paper_corners[0])
        print("右上：", paper_corners[1])
        print("右下：", paper_corners[2])
        print("左下：", paper_corners[3])
    except Exception as e:
        print("执行失败：", e)