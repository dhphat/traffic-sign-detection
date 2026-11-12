# Vietnamese Traffic Sign Detection with Ultralytics YOLO

Dự án huấn luyện và đánh giá mô hình phát hiện biển báo giao thông Việt Nam sử dụng Ultralytics YOLO (phiên bản nano: `yolo26n`).

---

## 1. Thông tin Dataset

Dataset được lưu trữ trong thư mục `data/` với chuẩn định dạng YOLO:
* **Số lượng ảnh**:
  * **Train**: 2,118 ảnh
  * **Validation**: 196 ảnh
  * **Test**: 103 ảnh
* **6 lớp đối tượng**:
  1. `No U-turn` (Cấm quay đầu)
  2. `No going opposite` (Cấm đi ngược chiều)
  3. `No stopping` (Cấm dừng xe)
  4. `No waiting` (Cấm đỗ xe)
  5. `Roundabout ahead` (Giao nhau chạy theo vòng xuyến)
  6. `Speed limit 60kmh` (Tốc độ tối đa cho phép 60km/h)

File cấu hình dữ liệu: [`data/data.yaml`](data/data.yaml) (sử dụng đường dẫn tuyệt đối đến thư mục `data` và các thư mục `train/images`, `valid/images`, `test/images`).

---

## 2. Cài đặt môi trường

Cài đặt các thư viện cần thiết thông qua [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

> **Lưu ý**: Trên máy tính Mac Apple Silicon (M1/M2/M3/M4), PyTorch sẽ tự động sử dụng tăng tốc qua chip GPU `mps`.

---

## 3. Huấn luyện mô hình (`train.py`)

Kịch bản [`train.py`](train.py) thực hiện full fine-tune từ trọng số COCO pretrained với các siêu tham số tiêu chuẩn:
* **Optimizer**: `SGD`
* **Learning rate ban đầu (`lr0`)**: `0.01`
* **Batch size**: `32`
* **Kích thước ảnh (`imgsz`)**: `640`
* **Random seed cố định**: `42` (đảm bảo tính tái lập)
* **Model mặc định**: `yolo26n.pt` (có thể linh hoạt đổi sang `yolov8n.pt` hoặc `yolo11n.pt`)

### 3.1. Chạy mặc định (50 epochs)
```bash
python train.py
```

### 3.2. Tuỳ chỉnh số epoch qua dòng lệnh
```bash
# Huấn luyện 30 epoch
python train.py --epochs 30

# Huấn luyện 100 epoch với batch 16
python train.py --epochs 100 --batch 16
```

### 3.3. Các tham số dòng lệnh khác
| Tham số | Mặc định | Mô tả |
| :--- | :--- | :--- |
| `--epochs` | `50` | Số lượng epoch huấn luyện |
| `--batch` | `32` | Kích thước batch |
| `--imgsz` | `640` | Kích thước ảnh đầu vào |
| `--lr0` | `0.01` | Learning rate ban đầu |
| `--optimizer` | `SGD` | Thuật toán tối ưu (`SGD`, `Adam`, `AdamW`) |
| `--seed` | `42` | Random seed cố định |
| `--model` | `yolo26n.pt` | Trọng số khởi tạo COCO |
| `--data` | `data/data.yaml` | Đường dẫn file cấu hình dữ liệu |
| `--device` | Tự động | Thiết bị tính toán (`mps`, `0`, `cpu`) |

Trọng số tốt nhất sau khi huấn luyện sẽ được lưu tại:
`runs/detect/traffic_sign_train/weights/best.pt`

---

## 4. Đánh giá mô hình trên tập test (`eval.py`)

Kịch bản [`eval.py`](eval.py) đánh giá mô hình trên tập **test** (103 ảnh) tại cả 2 độ phân giải: **640** và **960**. Kết quả sẽ được ghi vào file [`metrics.json`](metrics.json).

### 4.1. Chạy đánh giá
```bash
python eval.py --weights runs/detect/traffic_sign_train/weights/best.pt
```

### 4.2. Các chỉ số được ghi lại trong `metrics.json`
* `mAP50`: Mean Average Precision tại IoU = 0.50.
* `mAP50_95`: Mean Average Precision tại IoU từ 0.50 đến 0.95.
* `precision`: Độ chính xác trung bình (Precision).
* `recall`: Độ phủ trung bình (Recall).
* `per_class_mAP`: Chi tiết mAP từng lớp theo tên cụ thể (No U-turn, Speed limit 60kmh, ...).
* `FPS`: Tốc độ xử lý khung hình/giây (tính cả preprocess + inference + postprocess).
* `FPS_inference_only`: Tốc độ xử lý chỉ tính riêng bước inference.
* `latency_ms`: Chi tiết thời gian xử lý mỗi ảnh theo miligiây.

---

## 5. Phát hiện biển báo trên video (`infer_video.py`)

Kịch bản [`infer_video.py`](infer_video.py) thực hiện nhận diện biển báo trên từng khung hình video và xuất video kết quả có vẽ bounding box, nhãn và độ tin cậy (confidence).

### 5.1. Chạy cơ bản
```bash
python infer_video.py --source /duong/dan/video_input.mp4 --weights runs/detect/traffic_sign_train/weights/best.pt
```

### 5.2. Chạy với các tham số tuỳ biến
```bash
python infer_video.py \
  --source sample_traffic.mp4 \
  --weights runs/detect/traffic_sign_train/weights/best.pt \
  --output runs/detect/traffic_result.mp4 \
  --conf 0.3 \
  --iou 0.45 \
  --imgsz 640
```

> **Mẹo**: Thêm cờ `--show` nếu muốn hiển thị cửa sổ trực tiếp trong khi video đang được xử lý (nhấn `q` để thoát).

---

## 6. Cấu trúc thư mục dự án

```text
traffic-sign/
├── data/
│   ├── data.yaml            # File cấu hình tập train/val/test và 6 lớp
│   ├── train/               # 2,118 ảnh và nhãn huấn luyện
│   ├── valid/               # 196 ảnh và nhãn kiểm thử trong khi train
│   └── test/                # 103 ảnh và nhãn đánh giá cuối cùng
├── train.py                 # Huấn luyện mô hình YOLO (SGD, lr0=0.01, seed=42)
├── eval.py                  # Đánh giá tập test ở imgsz 640 và 960 -> metrics.json
├── infer_video.py           # Suy luận trên video và xuất video có bounding box
├── requirements.txt         # Danh sách thư viện phụ thuộc
├── metrics.json             # Kết quả đánh giá (sau khi chạy eval.py)
└── README.md                # Tài liệu hướng dẫn sử dụng
```

