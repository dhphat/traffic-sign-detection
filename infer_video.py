"""Script chạy mô hình YOLO phát hiện biển báo trên video và lưu video kết quả có bounding box."""

import argparse
import sys
import time
from pathlib import Path
import cv2
import torch
from ultralytics import YOLO


def select_default_device() -> str:
    """Tự động lựa chọn thiết bị khả dụng (CUDA, Apple Silicon MPS hoặc CPU)."""
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy phát hiện biển báo giao thông trên video và lưu video kết quả."
    )
    parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Đường dẫn file video đầu vào.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/traffic_sign_train/weights/best.pt",
        help="Đường dẫn file trọng số YOLO (mặc định: runs/detect/traffic_sign_train/weights/best.pt).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="runs/detect/output_detected.mp4",
        help="Đường dẫn lưu video kết quả (mặc định: runs/detect/output_detected.mp4).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Ngưỡng confidence để lọc box (mặc định: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Ngưỡng NMS IoU (mặc định: 0.45).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước ảnh suy luận (mặc định: 640).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị suy luận: '0', 'cpu', 'mps' (mặc định: tự động chọn).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Hiển thị video realtime trong khi xử lý.",
    )
    return parser.parse_args()


def process_video(
    source_path: Path,
    output_path: Path,
    model: YOLO,
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    show: bool = False,
) -> None:
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise IOError(f"Không thể mở file video: {source_path}")

    # Thu thập metadata video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:
        fps = 25.0  # fallback FPS mặc định nếu không đọc được
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    print(f"Bắt đầu xử lý video: {source_path}")
    print(f"Độ phân giải: {width}x{height} | FPS: {fps:.2f} | Tổng số frame: {total_frames}")

    frame_count = 0
    start_time = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Dự đoán bounding box
            results = model.predict(
                source=frame,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                verbose=False,
            )

            # Vẽ bounding box lên frame
            annotated_frame = results[0].plot()

            # Ghi vào video đầu ra
            writer.write(annotated_frame)

            if show:
                cv2.imshow("Traffic Sign Detection", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\nĐã nhận lệnh dừng từ người dùng.")
                    break

            # Tiến độ xử lý
            if total_frames > 0 and (frame_count % 30 == 0 or frame_count == total_frames):
                elapsed = time.time() - start_time
                current_fps = frame_count / elapsed if elapsed > 0 else 0
                percent = (frame_count / total_frames) * 100
                print(
                    f"\rTiến độ: {frame_count}/{total_frames} frames ({percent:.1f}%) "
                    f"- Tốc độ: {current_fps:.1f} FPS",
                    end="",
                    flush=True,
                )
    finally:
        cap.release()
        writer.release()
        if show:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    print(f"\n\nHoàn thành! Đã xử lý {frame_count} frames trong {total_time:.2f}s ({avg_fps:.1f} FPS).")
    print(f"Video đã được lưu tại: {output_path.resolve()}")


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Không tìm thấy video đầu vào tại '{source_path}'.")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(
            f"Cảnh báo: Trọng số '{weights_path}' chưa tồn tại trên đĩa. "
            f"YOLO sẽ thử tải hoặc kiểm tra kiến trúc model..."
        )

    device = args.device if args.device else select_default_device()
    output_path = Path(args.output)

    print("=" * 60)
    print("Khởi chạy phát hiện biển báo trên video")
    print(f"- Video đầu vào: {source_path}")
    print(f"- Video đầu ra : {output_path}")
    print(f"- Trọng số model: {weights_path}")
    print(f"- Confidence   : {args.conf}")
    print(f"- IoU          : {args.iou}")
    print(f"- Kích thước   : {args.imgsz}")
    print(f"- Thiết bị     : {device}")
    print("=" * 60)

    model = YOLO(str(weights_path))
    process_video(
        source_path=source_path,
        output_path=output_path,
        model=model,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=device,
        show=args.show,
    )


if __name__ == "__main__":
    main()

