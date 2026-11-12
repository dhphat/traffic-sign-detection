"""Script đánh giá mô hình YOLO trên tập test tại các độ phân giải 640 và 960,
xuất các chỉ số mAP50, mAP50-95, precision, recall, mAP từng lớp và FPS ra file JSON.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict
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
        description="Đánh giá mô hình YOLO trên tập test ở các độ phân giải 640 và 960."
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="runs/detect/traffic_sign_train/weights/best.pt",
        help="Đường dẫn file trọng số model (mặc định: runs/detect/traffic_sign_train/weights/best.pt).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "data.yaml"),
        help="Đường dẫn tới file data.yaml (mặc định: data/data.yaml).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="metrics.json",
        help="Đường dẫn file JSON lưu kết quả đánh giá (mặc định: metrics.json).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị đánh giá: '0', 'cpu', 'mps' (mặc định: tự động chọn).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch size khi evaluate (mặc định: 16).",
    )
    parser.add_argument(
        "--imgsz-list",
        nargs="+",
        type=int,
        default=[640, 960],
        help="Danh sách kích thước ảnh cần test (mặc định: 640 960).",
    )
    return parser.parse_args()


def evaluate_at_imgsz(
    model: YOLO, data: str, imgsz: int, device: str, batch: int
) -> Dict[str, Any]:
    """Chạy validation trên tập test ở một mức imgsz cụ thể và thu thập các chỉ số."""
    print(f"\n---> Đang đánh giá tại imgsz = {imgsz}...")
    metrics = model.val(
        data=data,
        split="test",
        imgsz=imgsz,
        batch=batch,
        device=device,
        save_json=False,
        plots=False,
        verbose=False,
    )

    box = metrics.box
    names = metrics.names

    # Tốc độ và FPS
    speed = metrics.speed  # dict: {'preprocess': ms, 'inference': ms, 'loss': ms, 'postprocess': ms}
    prep_ms = speed.get("preprocess", 0.0)
    inf_ms = speed.get("inference", 0.0)
    post_ms = speed.get("postprocess", 0.0)
    total_latency_ms = prep_ms + inf_ms + post_ms

    fps_end_to_end = (1000.0 / total_latency_ms) if total_latency_ms > 0 else 0.0
    fps_inference = (1000.0 / inf_ms) if inf_ms > 0 else 0.0

    # mAP từng lớp
    per_class_map = {}
    for idx, class_name in names.items():
        class_idx = int(idx)
        class_map50_95 = float(box.maps[class_idx]) if class_idx < len(box.maps) else 0.0
        
        # Nếu có chi tiết AP50 cho từng lớp
        class_info = {
            "mAP50_95": round(class_map50_95, 4),
        }
        if hasattr(box, "all_ap") and box.all_ap is not None and len(box.all_ap) > class_idx:
            class_info["mAP50"] = round(float(box.all_ap[class_idx][0]), 4)

        per_class_map[class_name] = class_info

    result = {
        "imgsz": imgsz,
        "mAP50": round(float(box.map50), 4),
        "mAP50_95": round(float(box.map), 4),
        "precision": round(float(box.mp), 4),
        "recall": round(float(box.mr), 4),
        "FPS": round(fps_end_to_end, 2),
        "FPS_inference_only": round(fps_inference, 2),
        "latency_ms": {
            "preprocess": round(prep_ms, 3),
            "inference": round(inf_ms, 3),
            "postprocess": round(post_ms, 3),
            "total": round(total_latency_ms, 3),
        },
        "per_class_mAP": per_class_map,
    }

    print(
        f"Kết quả imgsz={imgsz}: "
        f"mAP50={result['mAP50']}, mAP50-95={result['mAP50_95']}, "
        f"Precision={result['precision']}, Recall={result['recall']}, "
        f"FPS={result['FPS']}"
    )
    return result


def main() -> None:
    args = parse_args()
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file trọng số tại '{weights_path}'. "
            "Vui lòng chạy train.py trước hoặc chỉ định --weights chính xác."
        )

    device = args.device if args.device else select_default_device()

    print("=" * 60)
    print("Bắt đầu đánh giá mô hình trên tập test")
    print(f"- Trọng số model: {weights_path}")
    print(f"- File dataset  : {args.data}")
    print(f"- Các mức imgsz : {args.imgsz_list}")
    print(f"- Thiết bị      : {device}")
    print(f"- File output   : {args.output}")
    print("=" * 60)

    model = YOLO(str(weights_path))

    evaluation_summary: Dict[str, Any] = {
        "model_weights": str(weights_path),
        "data_config": str(args.data),
        "device": device,
        "evaluations": {},
    }

    for imgsz in args.imgsz_list:
        res = evaluate_at_imgsz(
            model=model,
            data=args.data,
            imgsz=imgsz,
            device=device,
            batch=args.batch,
        )
        evaluation_summary["evaluations"][f"imgsz_{imgsz}"] = res

    # Lưu kết quả vào file JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_summary, f, indent=4, ensure_ascii=False)

    print(f"\nĐã lưu toàn bộ chỉ số đánh giá vào: {output_path.resolve()}")


if __name__ == "__main__":
    main()

