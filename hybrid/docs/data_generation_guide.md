# 📚 Hướng dẫn chi tiết: Quy trình tạo Data Training cho Hybrid HCVRP

> Tài liệu này giải thích từng bước cách dữ liệu được tạo, xử lý, và sử dụng trong quá trình training model Hybrid RRNCO-PARCO.

---

## Mục lục

1. [Tổng quan bài toán](#1-tổng-quan-bài-toán)
2. [Nguồn dữ liệu gốc — Dữ liệu thành phố RRNCO](#2-nguồn-dữ-liệu-gốc--dữ-liệu-thành-phố-rrnco)
3. [Chia Train/Test theo thành phố](#3-chia-traintest-theo-thành-phố)
4. [Quy trình tạo data (`generate_real_data.py`)](#4-quy-trình-tạo-data-generate_real_datapy)
5. [Cấu trúc data output (file `.npz`)](#5-cấu-trúc-data-output-file-npz)
6. [Cách data được sử dụng khi Training](#6-cách-data-được-sử-dụng-khi-training)
7. [Distance matrix trong Environment](#7-distance-matrix-trong-environment-hcvrprealenv_reset)
8. [Lệnh tạo data cụ thể](#8-lệnh-tạo-data-cụ-thể)
9. [Pipeline end-to-end](#9-pipeline-end-to-end)
10. [Giải thích các tham số quan trọng](#10-giải-thích-các-tham-số-quan-trọng)

---

## 1. Tổng quan bài toán

Project này giải bài toán **HCVRP (Heterogeneous Capacitated Vehicle Routing Problem)** — bài toán định tuyến xe **nhiều phương tiện không đồng nhất**, sử dụng **khoảng cách thực tế** (OSRM) thay vì khoảng cách Euclidean (đường chim bay).

```
Đầu vào:
  • 1 depot (kho hàng)
  • N khách hàng (mỗi người có demand/nhu cầu hàng hóa)
  • M xe (mỗi xe có capacity và speed KHÁC NHAU)

Mục tiêu:
  → Tối thiểu thời gian hoàn thành của xe CHẬM NHẤT (min-max time)
  → Tức là: minimize  max_i (tour_length_i / speed_i)
```

**Tại sao gọi là "Heterogeneous"?**
- Mỗi xe có **capacity** (sức chứa) khác nhau: `[20, 41]`
- Mỗi xe có **speed** (tốc độ) khác nhau: `[0.5, 1.0]`
- → Model phải học cách phân công xe thông minh dựa trên đặc tính của từng xe

---

## 2. Nguồn dữ liệu gốc — Dữ liệu thành phố RRNCO

### 2.1 OSRM là gì?

**OSRM** (Open Source Routing Machine) là engine tính đường đi thực trên bản đồ OpenStreetMap.
Thay vì tính khoảng cách đường chim bay `sqrt((x1-x2)² + (y1-y2)²)`, OSRM tính khoảng cách **đi theo đường thực** (qua các con đường, ngã tư, cầu...).

```
Euclidean (đường chim bay):     A ──────────── B  (5 km)

OSRM (đường thực):              A ──┐
                                     │  đường vòng qua
                                     │  sông, cao tốc...
                                     └──── B  (7.3 km)
```

> ⚠️ **Quan trọng:** Ma trận khoảng cách OSRM có thể **bất đối xứng** — đi từ A→B có thể khác đường so với B→A (do đường một chiều, cầu vượt...).

### 2.2 Cấu trúc dữ liệu thành phố

Dữ liệu gốc nằm ở `real-routing-nco/data/dataset/`, tổng cộng **101 thành phố** trên toàn thế giới:

```
real-routing-nco/data/dataset/
├── splited_cities_list.json          ← Danh sách train/test cities
├── Amsterdam/
│   └── Amsterdam_data.npz            ← Data của Amsterdam
├── Tokyo/
│   └── Tokyo_data.npz
├── London/
│   └── London_data.npz
├── ...                               ← (101 thành phố)
```

### 2.3 Bên trong mỗi file thành phố

Mỗi file `{city}_data.npz` chứa:

| Key | Shape | Kiểu dữ liệu | Ý nghĩa |
|-----|-------|--------------|---------|
| `points` | `(1000, 2)` | `float64` | Tọa độ GPS thực (latitude, longitude) của 1000 điểm |
| `distance` | `(1000, 1000)` | `float64` | Ma trận khoảng cách thực giữa 1000 điểm (đơn vị: km) |
| `duration` | `(1000, 1000)` | `float64` | Ma trận thời gian di chuyển (đơn vị: phút) |

**Ví dụ cụ thể — Amsterdam:**

```python
import numpy as np
data = np.load("Amsterdam/Amsterdam_data.npz")

data["points"].shape     # (1000, 2)
data["points"][0]        # [52.376, 4.879]  ← lat, lon thực của Amsterdam

data["distance"].shape   # (1000, 1000)
data["distance"].max()   # 6.30 km  ← khoảng cách xa nhất trong thành phố

data["duration"].shape   # (1000, 1000)
data["duration"].max()   # 15.34 phút
```

---

## 3. Chia Train/Test theo thành phố

File `splited_cities_list.json` chia 100 thành phố (trong 101) thành 2 tập:

| Tập | Số lượng | Ví dụ | Mục đích |
|-----|----------|-------|----------|
| **Train** | 80 thành phố | Moscow, Vienna, Auckland, Lagos, Hanoi, Rio... | Tạo val data + training sampling |
| **Test** | 20 thành phố | Tokyo, London, Rome, Mumbai, Osaka... | Tạo test data (out-of-distribution) |

```python
import orjson
with open("splited_cities_list.json", "r") as f:
    cities = orjson.loads(f.read())

len(cities["train"])  # 80
len(cities["test"])   # 20
```

> 💡 **Ý nghĩa:** Khi test, model phải **generalize sang thành phố chưa thấy bao giờ** — đây gọi là "out-of-distribution evaluation". Nếu model chỉ học thuộc pattern của 80 thành phố train mà không generalize, kết quả test sẽ kém.

---

## 4. Quy trình tạo data (`generate_real_data.py`)

Script `hybrid/scripts/generate_real_data.py` hỗ trợ **2 chế độ** tạo data.

### 4.1 Chế độ A: Tạo mới từ dữ liệu thành phố (`--data_dir`)

```
┌──────────────────────────────────────────┐
│  80 thành phố train (1000 điểm/city)     │
│  hoặc 20 thành phố test                  │
└───────────────────┬──────────────────────┘
                    │
       ┌────────────▼────────────────┐
       │  Bước 1: CHỌN THÀNH PHỐ    │
       │  Chia đều B instances cho   │
       │  tất cả cities              │
       │  VD: 1280/80 = 16/city      │
       └────────────┬────────────────┘
                    │
       ┌────────────▼─────────────────────┐
       │  Bước 2: LẤY MẪU 101 ĐIỂM       │
       │  Từ 1000 điểm của thành phố,     │
       │  chọn random 101 điểm            │
       │  (1 depot + 100 khách hàng)      │
       │                                   │
       │  2 cách lấy mẫu:                 │
       │  • uniform: random rải rác       │
       │  • cluster: chọn điểm gần nhau   │
       └────────────┬─────────────────────┘
                    │
       ┌────────────▼─────────────────────┐
       │  Bước 3: TRÍCH SUB-MATRIX        │
       │  Lấy ma trận khoảng cách 101×101 │
       │  tương ứng với 101 điểm đã chọn  │
       │                                   │
       │  distance[idx][:, idx]            │
       │  → từ 1000×1000 → 101×101        │
       └────────────┬─────────────────────┘
                    │
       ┌────────────▼─────────────────────┐
       │  Bước 4: NORMALIZE TỌA ĐỘ       │
       │  Min-max per instance → [0, 1]   │
       │                                   │
       │  points_norm = (p - p_min)       │
       │               / (p_max - p_min)  │
       │                                   │
       │  VD: [52.3, 4.8] → [0.0, 0.0]   │
       │      [52.4, 4.9] → [1.0, 1.0]   │
       └────────────┬─────────────────────┘
                    │
       ┌────────────▼──────────────────────┐
       │  Bước 5: SINH DỮ LIỆU MULTI-AGENT│
       │  (Dữ liệu RRNCO chỉ có 1 xe,    │
       │   cần thêm thông tin cho 7 xe)    │
       │                                    │
       │  • demand: random [1, 10] × 100   │
       │  • capacity: random [20, 41] × 7  │
       │  • speed: random [0.5, 1.0] × 7   │
       │  • num_agents: 7                   │
       └────────────┬──────────────────────┘
                    │
       ┌────────────▼──────────────────┐
       │  Bước 6: LƯU FILE .npz       │
       │  np.savez(output_path, **data)│
       └───────────────────────────────┘
```

#### Chi tiết Bước 2 — Hai cách lấy mẫu:

**Uniform sampling** (mặc định):
```python
# Chọn ngẫu nhiên 101 điểm từ 1000 → phân bố rải rác
idx = rng.choice(1000, 101, replace=False)
```

```
  Thành phố (1000 điểm)          Instance (101 điểm)
  ·  ·  ·  ·  ·  ·  ·            ·     ·     ·
  ·  ·  ·  ·  ·  ·  ·      →       ·        ·
  ·  ·  ·  ·  ·  ·  ·            ·  ·     ·
  ·  ·  ·  ·  ·  ·  ·               ·  ·
```

**Cluster sampling**:
```python
# Chọn 1 điểm trung tâm, lấy 100 điểm gần nhất → tập trung 1 vùng
center_idx = rng.integers(1000)
dists = np.linalg.norm(points - points[center_idx], axis=1)
idx = np.argsort(dists)[:101]
```

```
  Thành phố (1000 điểm)          Instance (101 điểm)
  ·  ·  ·  ·  ·  ·  ·
  ·  ·  ·  ·  ·  ·  ·      →    ····
  ·  ·  ·  ·  ·  ·  ·           ·····  (tập trung 1 vùng)
  ·  ·  ·  ·  ·  ·  ·           ····
```

### 4.2 Chế độ B: Chuyển đổi file RRNCO có sẵn (`--from_npz`)

Khi đã có file `.npz` từ RRNCO (single-vehicle CVRP), chỉ cần **thêm thông tin multi-agent**:

```
RRNCO .npz (1 xe)                    Hybrid .npz (7 xe)
┌────────────────────────┐           ┌─────────────────────────┐
│ depot    [B, 2]        │──────────▶│ depot    [B, 2]         │  (giữ nguyên)
│ locs     [B, 100, 2]  │──────────▶│ locs     [B, 100, 2]    │  (giữ nguyên)
│ demand   [B, 100]     │──────────▶│ demand   [B, 100]       │  (giữ nguyên)
│ distance_matrix        │──────────▶│ distance_matrix          │  (giữ nguyên)
│         [B, 101, 101]  │          │         [B, 101, 101]    │
│ capacity (scalar: 50)  │          │                          │
│                        │  THÊM MỚI│ capacity [B, 7]  ← rand │  (random mỗi xe)
│                        │──────────▶│ speed    [B, 7]  ← rand │  (random mỗi xe)
│                        │          │ num_agents [B]   ← 7    │  (cố định = 7)
└────────────────────────┘           └─────────────────────────┘
```

### 4.3 Xử lý outlier (điểm bất thường)

Một số điểm trong city data có khoảng cách OSRM > 100,000 km — đây là lỗi routing (OSRM không tìm được đường). Script tự động loại bỏ:

```python
# Nếu bất kỳ khoảng cách nào > 1e5, đánh dấu là outlier
if distance.max() > 1e5:
    outlier_mask = np.any(distance > 1e5, axis=1) | np.any(distance > 1e5, axis=0)
    keep = np.where(~outlier_mask)[0]
    # Loại bỏ các điểm outlier khỏi points và distance matrix
    points = points[keep]
    distance = distance[keep][:, keep]
```

---

## 5. Cấu trúc data output (file `.npz`)

### 5.1 Các key trong file `.npz`

| Key | Shape | Dtype | Ý nghĩa | Phạm vi giá trị |
|-----|-------|-------|---------|-----------------|
| `depot` | `[B, 2]` | `float32` | Tọa độ depot (đã normalize) | `[0, 1]` |
| `locs` | `[B, N, 2]` | `float32` | Tọa độ N khách hàng (đã normalize) | `[0, 1]` |
| `demand` | `[B, N]` | `float32` | Nhu cầu hàng hóa mỗi khách hàng | `[1, 10]` integer |
| `capacity` | `[B, M]` | `float32` | Sức chứa mỗi xe | `[20, 41]` float |
| `speed` | `[B, M]` | `float32` | Tốc độ mỗi xe | `[0.5, 1.0]` float |
| `distance_matrix` | `[B, N+1, N+1]` | `float32` | Ma trận khoảng cách OSRM | km (raw, chưa normalize) |
| `num_agents` | `[B]` | `int64` | Số xe | Cố định = M |

Trong đó: `B` = batch size (số instances), `N` = 100 (số khách hàng), `M` = 7 (số xe).

### 5.2 Ví dụ thực tế

**File `val_real_n100_m7.npz`:**
```
depot:           (1248, 2)      float32   min=0.00  max=1.00
locs:            (1248, 100, 2) float32   min=0.00  max=1.00
demand:          (1248, 100)    float32   min=1.00  max=10.00
capacity:        (1248, 7)      float32   min=20.00 max=41.00
speed:           (1248, 7)      float32   min=0.50  max=1.00
distance_matrix: (1248, 101, 101) float32 min=0.00  max=12.06
num_agents:      (1248,)        int64     min=7     max=7
```

> 📝 **Tại sao 1248 mà không phải 1280?** Vì `dataset_size=1280` chia đều cho 80 thành phố = 16/city, nhưng một số thành phố có thể bị skip do thiếu điểm → `1248 = 78 cities × 16`.

### 5.3 Ý nghĩa distance_matrix

```
distance_matrix[i] = ma trận 101×101 cho instance thứ i

         depot  c1    c2    c3   ... c100
depot  [  0.0   2.3   1.5   3.7  ... 4.2  ]
c1     [  2.1   0.0   0.8   2.9  ... 3.5  ]   ← Lưu ý: 2.3 ≠ 2.1
c2     [  1.6   0.9   0.0   1.2  ... 2.8  ]     (bất đối xứng!)
c3     [  3.5   2.7   1.3   0.0  ... 1.9  ]
...
c100   [  4.0   3.6   2.9   2.0  ... 0.0  ]
```

> ⚠️ `distance_matrix[i][a][b] ≠ distance_matrix[i][b][a]` — đây là đặc trưng của khoảng cách đường thực (đường 1 chiều, cầu vượt...).

---

## 6. Cách data được sử dụng khi Training

### 6.1 Hai chế độ training

Project có 2 config training:

| Config | Generator | Training data | Khi nào dùng |
|--------|-----------|--------------|-------------|
| `hybrid_hcvrp_real.yaml` | `HCVRPRealGenerator` | **Euclidean on-the-fly** | Mặc định, nhanh |
| `hybrid_hcvrp_real_osrm.yaml` | `HCVRPRealWorldGenerator` | **OSRM on-the-fly** | Khi muốn train trên data thực |

### 6.2 Chế độ mặc định — Euclidean training + Real-world val/test

```
┌─────────────────────────────────────────────────────────────┐
│  TRAINING (on-the-fly, không cần file)                      │
│                                                              │
│  HCVRPRealGenerator:                                         │
│  1. Random tọa độ depot, locs trong [0, 1]                  │
│  2. Tính distance_matrix = Euclidean distance               │
│     (torch.cdist)                                            │
│  3. Random demand, capacity, speed                           │
│                                                              │
│  → Vô hạn data, đa dạng, nhanh                             │
│  → Mỗi epoch sinh data mới hoàn toàn                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  VALIDATION (load file .npz)                                │
│                                                              │
│  Load: data/hcvrp_real/val_real_n100_m7.npz                 │
│  → 1248 instances từ 80 train cities (in-distribution)      │
│  → Khoảng cách OSRM thực                                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  TEST (load file .npz)                                      │
│                                                              │
│  Load: data/hcvrp_real/test_real_n100_m7.npz                │
│  → 1280 instances từ 20 test cities (out-of-distribution)   │
│  → Khoảng cách OSRM thực                                   │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Chế độ OSRM — Real-world training

Khi dùng `hybrid_hcvrp_real_osrm.yaml`:

```
TRAINING (on-the-fly từ city data trong RAM):
  1. Random chọn 1 thành phố (từ 80 train cities)
  2. Random chọn 101 điểm từ thành phố đó
  3. Trích sub-matrix khoảng cách OSRM 101×101
  4. Normalize tọa độ → [0, 1]
  5. Random demand, capacity, speed
  → Mỗi batch là data OSRM thực!
```

### 6.4 Curriculum sampling (kỹ thuật training nâng cao)

Config cho phép **thay đổi kích thước bài toán** trong quá trình training:

```yaml
# Trong hybrid_hcvrp_real.yaml:
train_min_agents: 3          # Bắt đầu với ít xe
train_max_agents: 7          # Tăng dần lên 7 xe
train_min_size: 60           # Bắt đầu với 60 khách hàng
train_max_size: 100          # Tăng dần lên 100 khách hàng
```

→ Model học từ bài toán nhỏ → lớn, giúp **hội tụ tốt hơn**.

---

## 7. Distance matrix trong Environment (`HCVRPRealEnv._reset`)

### 7.1 Tại sao cần mở rộng distance matrix?

Data gốc có distance matrix `[B, 101, 101]` — index 0 = depot, index 1-100 = customers.

Nhưng trong HCVRP multi-agent, **mỗi xe xuất phát từ depot** → cần **7 "bản sao" của depot**:

```
Raw matrix [101×101]:                 Expanded matrix [107×107]:
                                      
     depot  c1  c2 ... c100              a1  a2 ... a7  c1  c2 ... c100
depot [ 0   d01 d02 ... d0,100]     a1  [ 0   0  ... 0  d01 d02 ... d0,100]
c1   [d10   0   d12 ... d1,100]     a2  [ 0   0  ... 0  d01 d02 ... d0,100]
c2   [d20  d21   0  ... d2,100]     ... 
...                                 a7  [ 0   0  ... 0  d01 d02 ... d0,100]
c100 [d100,0 ............  0  ]     c1  [d10 d10 ...d10  0   d12 ... d1,100]
                                    c2  [d20 d20 ...d20 d21   0  ... d2,100]
                                    ...
                                    c100[................              0    ]
```

### 7.2 Code mở rộng

```python
# Trong HCVRPRealEnv._reset():

# raw_distance: [B, 101, 101]  (1 depot + 100 customers)
depot_row = raw_distance[:, :1, :]           # [B, 1, 101]
customer_rows = raw_distance[:, 1:, :]       # [B, 100, 101]

# Lặp depot row cho 7 xe
depot_rows_expanded = depot_row.repeat(1, 7, 1)  # [B, 7, 101]

# Ghép: [B, 107, 101]
expanded_rows = torch.cat([depot_rows_expanded, customer_rows], dim=1)

# Làm tương tự cho columns → [B, 107, 107]
```

### 7.3 Normalize distance matrix

Sau khi mở rộng, distance matrix được **normalize per-instance** về `[0, 1]`:

```python
min_dist = distance_matrix.amin(dim=(-2, -1))  # min của toàn bộ matrix
max_dist = distance_matrix.amax(dim=(-2, -1))  # max của toàn bộ matrix

distance_matrix_norm = (distance_matrix - min_dist) / (max_dist - min_dist + 1e-6)
```

> `min_dist` và `max_dist` được lưu lại trong TensorDict để có thể **denormalize** khi cần.

---

## 8. Lệnh tạo data cụ thể

### 8.1 Tạo validation set (in-distribution)

```bash
cd hybrid

python scripts/generate_real_data.py \
    --data_dir ../real-routing-nco/data/dataset \
    --output_dir data/hcvrp_real \
    --graph_size 100 \
    --dataset_size 1280 \
    --num_agents 7 \
    --split train \
    --seed 42 \
    --output_name val_real_n100_m7.npz
```

**Giải thích:**
- `--data_dir`: Thư mục chứa dữ liệu thành phố RRNCO
- `--split train`: Lấy từ 80 thành phố train → **in-distribution**
- `--graph_size 100`: Mỗi instance có 100 khách hàng
- `--dataset_size 1280`: Tạo 1280 instances
- `--num_agents 7`: 7 xe
- `--seed 42`: Seed random để reproduce

### 8.2 Tạo test set (out-of-distribution)

```bash
python scripts/generate_real_data.py \
    --data_dir ../real-routing-nco/data/dataset \
    --output_dir data/hcvrp_real \
    --graph_size 100 \
    --dataset_size 1280 \
    --num_agents 7 \
    --split test \
    --seed 42 \
    --output_name test_real_n100_m7.npz
```

**Khác biệt duy nhất:** `--split test` → Lấy từ 20 thành phố test → **out-of-distribution**

### 8.3 Chuyển đổi file RRNCO có sẵn

```bash
python scripts/generate_real_data.py \
    --from_npz ../real-routing-nco/data/rcvrp/rcvrp_n100_seed3333_in_distribution.npz \
    --output_dir data/hcvrp_real \
    --num_agents 7 \
    --seed 42
```

Output tự động đặt tên: `hcvrp_rcvrp_n100_seed3333_in_distribution_m7.npz`

### 8.4 Tạo data với cluster sampling

```bash
python scripts/generate_real_data.py \
    --data_dir ../real-routing-nco/data/dataset \
    --output_dir data/hcvrp_real \
    --graph_size 100 \
    --dataset_size 1280 \
    --num_agents 7 \
    --split train \
    --dist_type cluster \
    --seed 42 \
    --output_name val_cluster_n100_m7.npz
```

---

## 9. Pipeline end-to-end

```
┌─────────────────────────────────────────────────────────────────┐
│                    PHASE 1: DATA NGUỒN                          │
│                                                                  │
│  OpenStreetMap → OSRM (Docker container)                        │
│       ↓                                                          │
│  Tính distance matrix cho 1000 điểm/thành phố                  │
│       ↓                                                          │
│  101 thành phố × {points, distance, duration}                   │
│  Lưu tại: real-routing-nco/data/dataset/{city}/{city}_data.npz  │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    PHASE 2: CHIA TRAIN/TEST                      │
│                                                                  │
│  splited_cities_list.json:                                       │
│  ├── train: 80 thành phố (Moscow, Vienna, Auckland, ...)        │
│  └── test:  20 thành phố (Tokyo, London, Rome, ...)             │
└──────────────────────────────┬──────────────────────────────────┘
                               │
              ┌────────────────┴─────────────────┐
              │                                  │
┌─────────────▼──────────────┐  ┌────────────────▼─────────────┐
│  80 TRAIN CITIES           │  │  20 TEST CITIES              │
│                            │  │                              │
│  Sample 101 điểm/instance  │  │  Sample 101 điểm/instance   │
│  + demand, capacity, speed │  │  + demand, capacity, speed   │
│                            │  │                              │
│  → val_real_n100_m7.npz   │  │  → test_real_n100_m7.npz    │
│    (1248 instances)        │  │    (1280 instances)          │
│    IN-DISTRIBUTION         │  │    OUT-OF-DISTRIBUTION       │
└─────────────┬──────────────┘  └────────────────┬─────────────┘
              │                                  │
              └────────────────┬─────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    PHASE 3: TRAINING                             │
│                                                                  │
│  Training loop (each epoch):                                     │
│  ├── Train:  HCVRPRealGenerator sinh Euclidean data on-the-fly  │
│  │           (hoặc HCVRPRealWorldGenerator sinh OSRM data)      │
│  ├── Val:    Load val_real_n100_m7.npz (OSRM data)             │
│  └── Test:   Load test_real_n100_m7.npz (OSRM data)            │
└──────────────────────────────┬──────────────────────────────────┘
                               │
┌──────────────────────────────▼──────────────────────────────────┐
│                    PHASE 4: ENVIRONMENT                          │
│                                                                  │
│  HCVRPRealEnv._reset():                                         │
│  1. Load/Generate TensorDict with distance_matrix [B, 101, 101] │
│  2. Expand depot → agents: [B, 107, 107]                        │
│  3. Normalize per-instance: [0, 1]                               │
│  4. Feed to Encoder (RRNCO-style) + Decoder (PARCO-style)       │
│                                                                  │
│  Reward = - max_agent(tour_length / speed)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. Giải thích các tham số quan trọng

### 10.1 Tham số data generation

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `graph_size` | 100 | Số khách hàng (N). File output có N+1 nodes (thêm depot) |
| `dataset_size` | 1280 | Số instances trong dataset |
| `num_agents` | 7 | Số xe (M) |
| `split` | `train` | Dùng train/test cities |
| `dist_type` | `uniform` | Cách lấy mẫu điểm: `uniform` hoặc `cluster` |
| `seed` | 42 | Random seed để reproduce kết quả |

### 10.2 Tham số multi-agent

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `min_demand` / `max_demand` | 1 / 10 | Nhu cầu hàng hóa mỗi khách hàng |
| `min_capacity` / `max_capacity` | 20 / 41 | Sức chứa mỗi xe (random uniform) |
| `min_speed` / `max_speed` | 0.5 / 1.0 | Tốc độ mỗi xe (random uniform) |

### 10.3 Tham số environment (config YAML)

```yaml
# configs/env/hcvrp_real.yaml
_target_: hybrid.envs.hcvrp_real.HCVRPRealEnv
normalize_distance: true          # Normalize distance matrix về [0, 1]

generator_params:
  num_loc: 100                    # = graph_size
  num_agents: 7                   # = num_agents
  compute_distance_matrix: true   # Tính Euclidean matrix khi training
```

### 10.4 Mối quan hệ giữa capacity và demand

```
Tổng demand trung bình = N × (min_demand + max_demand) / 2
                        = 100 × (1 + 10) / 2 = 550

Tổng capacity trung bình = M × (min_capacity + max_capacity) / 2
                          = 7 × (20 + 41) / 2 = 213.5

→ Tỷ lệ demand/capacity ≈ 550/213.5 ≈ 2.6
→ Mỗi xe trung bình phải đi 2-3 chuyến (quay về depot nạp hàng)
```

---

## Kiểm tra data đã tạo

```python
import numpy as np

# Load và kiểm tra
data = np.load("data/hcvrp_real/val_real_n100_m7.npz")

print("Keys:", data.files)
for key in data.files:
    arr = data[key]
    print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}, "
          f"min={arr.min():.4f}, max={arr.max():.4f}")

# Kiểm tra distance matrix có bất đối xứng không
dm = data["distance_matrix"]  # [B, 101, 101]
asymmetry = np.abs(dm - dm.transpose(0, 2, 1)).mean()
print(f"\nAsymmetry trung bình: {asymmetry:.4f} km")
# Nếu > 0 → ma trận bất đối xứng (khoảng cách đường thực)
```

---

> 📌 **Tóm tắt 1 câu:** Data training được tạo bằng cách lấy mẫu các sub-problem (101 điểm + distance matrix OSRM) từ 101 thành phố thực trên thế giới, sau đó thêm thông tin multi-agent (capacity, speed) để chuyển từ bài toán CVRP 1 xe → HCVRP 7 xe.
