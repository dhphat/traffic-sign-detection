"""Script huấn luyện mô hình YOLO phát hiện biển báo giao thông."""

import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO


def set_seed(seed: int = 42) -> None:
    """Cố định seed để đảm bảo tính tái lập (reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def select_default_device() -> str:
    """Tự động lựa chọn thiết bị khả dụng (CUDA, Apple Silicon MPS hoặc CPU)."""
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Huấn luyện mô hình YOLO phát hiện biển báo giao thông."
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Số lượng epoch huấn luyện (mặc định: 50).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size (mặc định: 32).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Kích thước ảnh đầu vào (mặc định: 640).",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.01,
        help="Learning rate ban đầu cho SGD (mặc định: 0.01).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="SGD",
        choices=["SGD", "Adam", "AdamW"],
        help="Thuật toán tối ưu (mặc định: SGD).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Giá trị random seed (mặc định: 42).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo26n.pt",
        help="Trọng số khởi tạo COCO hoặc kiến trúc YOLO (mặc định: yolo26n.pt).",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(Path(__file__).resolve().parent / "data" / "data.yaml"),
        help="Đường dẫn tới file cấu hình dataset (mặc định: data/data.yaml).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/detect",
        help="Thư mục lưu kết quả huấn luyện (mặc định: runs/detect).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="traffic_sign_train",
        help="Tên thư mục run cụ thể (mặc định: traffic_sign_train).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị huấn luyện: '0', 'cpu', 'mps' (mặc định: tự động chọn).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Số lượng worker cho dataloader (mặc định: 4).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = args.device if args.device else select_default_device()

    print("=" * 60)
    print("Bắt đầu huấn luyện mô hình YOLO phát hiện biển báo giao thông")
    print(f"- Model khởi tạo   : {args.model}")
    print(f"- File dữ liệu     : {args.data}")
    print(f"- Epochs           : {args.epochs}")
    print(f"- Batch size       : {args.batch}")
    print(f"- Kích thước ảnh   : {args.imgsz}")
    print(f"- Optimizer        : {args.optimizer}")
    print(f"- Learning rate lr0: {args.lr0}")
    print(f"- Random seed      : {args.seed}")
    print(f"- Thiết bị         : {device}")
    print("=" * 60)

    # Khởi tạo mô hình từ trọng số COCO
    model = YOLO(args.model)

    # Full fine-tune
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        optimizer=args.optimizer,
        lr0=args.lr0,
        seed=args.seed,
        deterministic=True,
        device=device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        exist_ok=True,
    )

    print("\nHuấn luyện hoàn tất!")
    save_dir = Path(args.project) / args.name
    print(f"Trọng số tốt nhất đã lưu tại: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()

