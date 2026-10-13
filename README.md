# 🎬 Movie Data Analysis & Prediction Web App (IMDb/TMDB)

Tài liệu hướng dẫn thi công dự án dành cho các tổ thành viên.
Đọc kỹ mục **Quy ước chung** trước khi bắt tay vào code.

---

## 1. Cấu trúc dự án

```text
phimro-database/
├── data/
│   ├── results.csv          # Dữ liệu đầu vào (Tổ 1 / mock data)
│   └── results2.csv         # Bảng thống kê xuất ra (Tổ 2)
├── src/
│   ├── data_pipeline.py     # Phần I  - Làm sạch dữ liệu
│   ├── statistics_eda.py    # Phần II - Thống kê mô tả
│   ├── models.py            # Phần III & IV - Học máy
│   └── visualizer.py        # Phần II & III - Biểu đồ
├── app.py                   # Giao diện Streamlit (Nhóm trưởng phụ trách)
├── requirements.txt         # Danh sách thư viện
└── README.md                # Tài liệu này
```

---

## 2. Cài đặt và chạy dự án

Yêu cầu: Python 3.10 trở lên.

```powershell
Set-Location "d:\PRJ\phimro-database"
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Mở trình duyệt tại `http://localhost:8501` để xem giao diện.

> ⚠️ Nếu gặp lỗi phiên bản scikit-learn khi chạy,
> hãy chạy lại `python -m pip install -r requirements.txt`.

---

## 3. Quy ước chung TRƯỚC KHI CODE

### 3.1. Dữ liệu đầu vào (Mock Data)

Trong lúc chờ **Tổ 1** thu thập dữ liệu thật, mỗi thành viên tự tạo
file `data/results.csv` giả lập (hoặc lấy bản chung của tổ) với các cột
chuẩn sau:

| Cột | Kiểu dữ liệu | Ví dụ | Ghi chú |
|---|---|---|---|
| `Title` | str | `Inception` | Tên phim, dùng làm khóa tìm kiếm |
| `Genre` | str | `Sci-Fi` | Dùng cho groupby ở Phần II |
| `IMDb_Rating` | float | `8.8` | Thang 0–10 |
| `Budget_USD` | int | `160000000` | Đơn vị: USD |
| `Revenue_USD` | int | `836800000` | Đơn vị: USD |
| `Votes` | int | `2300000` | Bắt buộc cho biểu đồ radar (Phần III) |
| `Duration_min` | int | `148` | Bắt buộc cho biểu đồ radar (Phần III) |

Gợi ý tạo nhanh dữ liệu giả lập (chạy 1 lần tại thư mục gốc dự án):

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
genres = ["Action", "Comedy", "Drama", "Sci-Fi", "Horror", "Animation"]
rows = []
for i in range(60):
    budget = int(rng.integers(1_000_000, 250_000_000))
    rows.append({
        "Title": f"Mock Movie {i + 1}",
        "Genre": rng.choice(genres),
        "IMDb_Rating": round(float(rng.uniform(2.0, 9.5)), 1),
        "Budget_USD": budget,
        "Revenue_USD": int(budget * rng.uniform(0.2, 6.0)),
        "Votes": int(rng.integers(1_000, 2_000_000)),
        "Duration_min": int(rng.integers(80, 190)),
    })

pd.DataFrame(rows).to_csv("data/results.csv", index=False)
```

### 3.2. Nguyên tắc code

1. **Viết code trực tiếp vào các hàm đã được Nhóm trưởng định nghĩa sẵn**
   trong thư mục `src/`. Thay chữ `pass` bằng phần cài đặt của mình.
2. **Tuyệt đối không thay đổi tên hàm** hay tham số đầu vào/đầu ra.
   Nếu cần tham số mới, phải có giá trị mặc định và báo Nhóm trưởng
   trước.
3. **Không sửa `app.py`** — Nhóm trưởng là người duy nhất import hàm
   vào giao diện và thiết kế nút bấm.
4. Giữ nguyên **docstring tiếng Việt**; bổ sung ghi chú của mình vào
   cuối docstring nếu cần.
5. Thư viện sử dụng chính: `pandas` (xử lý bảng),
   `matplotlib.pyplot` (vẽ hình), `sklearn` (học máy).
6. Chạy mọi lệnh test **từ thư mục gốc dự án** để đường dẫn tương đối
   `data/results.csv`, `data/results2.csv` hoạt động đúng.
7. Không commit dữ liệu lớn; file `*.csv` trong `data/` chỉ dùng nội bộ.

---

## 4. Bảng hợp đồng hàm (Contract) — tra cứu nhanh

| Hàm | File | Trả về |
|---|---|---|
| `load_and_clean_data(raw_file_path)` | `src/data_pipeline.py` | `DataFrame` đã làm sạch |
| `calculate_basic_stats(df)` | `src/statistics_eda.py` | `DataFrame` thống kê + xuất `data/results2.csv` |
| `get_top_bottom_movies(df, column, top_n=3)` | `src/statistics_eda.py` | `(top_movies, bottom_movies)` |
| `plot_histogram(df, column)` | `src/visualizer.py` | `matplotlib.figure.Figure` |
| `run_kmeans_clustering(df, n_clusters)` | `src/models.py` | `DataFrame` có cột `Cluster` |
| `run_pca_reduction(df)` | `src/models.py` | `DataFrame` tọa độ PCA 2 cột |
| `train_revenue_prediction_model(df)` | `src/models.py` | `Pipeline` sklearn đã huấn luyện |
| `plot_radar_chart(df, movie1_name, movie2_name)` | `src/visualizer.py` | `matplotlib.figure.Figure` |

**Quy tắc chung khi trả về hình:** trả về đối tượng `fig`, **không**
gọi `plt.show()`; bên `app.py` sẽ dùng `st.pyplot(fig)` và đóng hình.

---

## 5. Phân công chi tiết theo tổ

### 📌 TỔ 1 — Phần I: Thu thập & làm sạch dữ liệu

**File:** `src/data_pipeline.py`

- Thu thập dữ liệu thật (web scraping bằng `requests` +
  `beautifulsoup4` hoặc tải dataset mở) và thay file mock
  `data/results.csv`.
- Cài đặt `load_and_clean_data(raw_file_path)`:
  - Đọc CSV; chuyển `"N/a"`, chuỗi rỗng thành giá trị thiếu (`NaN`).
  - Loại khoảng trắng thừa, bỏ dòng trùng lặp theo `Title`.
  - Ép kiểu số cho `IMDb_Rating`, `Budget_USD`, `Revenue_USD`,
    `Votes`, `Duration_min`.
  - Trả về `DataFrame` sạch; lỗi tệp hoặc thiếu cột phải báo rõ.

**Nghiệm thu:** chạy
`python -c "from src.data_pipeline import load_and_clean_data; print(load_and_clean_data('data/results.csv').info())"`
không báo lỗi.

---

### 📌 TỔ 2 — Phần II: Thống kê & Trực quan hóa

**File chính:** `src/statistics_eda.py` và `src/visualizer.py`

#### Nhiệm vụ 1 — Tính thống kê và xuất `results2.csv`

Hàm: `calculate_basic_stats(df)`

- Tính **Median, Mean, Std** cho `IMDb_Rating`, `Revenue_USD`,
  `Budget_USD` trên **toàn bộ dữ liệu** (hàng `All`)
  và **nhóm theo `Genre`**.
- Xuất file: `df_results.to_csv("data/results2.csv", index=True)`
  (chạy từ thư mục gốc dự án).
- Trả về `df_results` để hiển thị lên web.

Gợi ý thuật toán:

```python
import pandas as pd

def calculate_basic_stats(df):
    cols = ["IMDb_Rating", "Revenue_USD", "Budget_USD"]

    # 1. Thống kê TOÀN BỘ dữ liệu (All)
    all_stats = df[cols].agg(["mean", "median", "std"]).T
    all_stats.index = pd.MultiIndex.from_product(
        [["All"], all_stats.index]
    )

    # 2. Thống kê theo TỪNG THỂ LOẠI (Groupby Genre)
    genre_stats = df.groupby("Genre")[cols].agg(["mean", "median", "std"])
    # ... sinh viên ghép 2 bảng trên thành df_results thống nhất ...

    # df_results.to_csv("data/results2.csv", index=True)
    return df_results  # Trả về bảng để hiển thị lên Web
```

#### Nhiệm vụ 2 — Top phim điểm cao nhất / thấp nhất

Hàm: `get_top_bottom_movies(df, column, top_n=3)`

```python
def get_top_bottom_movies(df, column, top_n=3):
    top_movies = df.nlargest(top_n, column)[["Title", column]]
    bottom_movies = df.nsmallest(top_n, column)[["Title", column]]
    return top_movies, bottom_movies
```

#### Nhiệm vụ 3 — Biểu đồ phân bố (Histogram)

Hàm: `plot_histogram(df, column)` trong `src/visualizer.py`

```python
import matplotlib.pyplot as plt

def plot_histogram(df, column):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(df[column].dropna(), bins=20,
            color="skyblue", edgecolor="black")
    ax.set_title(f"Phân bố của {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Số lượng phim")
    return fig  # Trả về fig để Streamlit hiển thị (st.pyplot(fig))
```

---

### 📌 TỔ 3 — Phần III: Học máy & Biểu đồ nâng cao

**File chính:** `src/models.py` và `src/visualizer.py`

#### Nhiệm vụ 1 — Phân cụm K-means

Hàm: `run_kmeans_clustering(df, n_clusters)`

Gom nhóm phim (bom tấn, phim xịt...) dựa trên `Budget_USD`,
`Revenue_USD`, `IMDb_Rating`.

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def run_kmeans_clustering(df, n_clusters):
    features = ["Budget_USD", "Revenue_USD", "IMDb_Rating"]
    X = df[features].dropna().copy()

    # RẤT QUAN TRỌNG: đưa tiền và điểm số về cùng thang đo
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    X["Cluster"] = kmeans.fit_predict(X_scaled)

    return X  # Bảng dữ liệu đã có cột 'Cluster' (nhãn 0, 1, 2...)
```

> 📝 **Ghi chú cho báo cáo:** chạy vòng lặp `n_clusters` từ 1–10, vẽ
> **Elbow Method** (biểu đồ inertia) để biện luận vì sao chọn K = 3
> hoặc K = 4.

#### Nhiệm vụ 2 — Giảm chiều PCA (bản đồ cụm 2D)

Hàm: `run_pca_reduction(df)`

> ⚠️ **Lưu ý chữ ký:** Hướng dẫn gốc gợi ý nhận `X_scaled` (mảng đã
> chuẩn hóa). Để thống nhất đầu vào `df` như hợp đồng hàm, **hàm tự
> chuẩn hóa ở bên trong** (dùng `StandardScaler` trước khi chạy PCA).
> Sinh viên nhận `df` rồi tự chọn cột số, drop dòng thiếu, scale, PCA.

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def run_pca_reduction(df):
    features = ["Budget_USD", "Revenue_USD", "IMDb_Rating"]
    X = df[features].dropna()

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(
        data=principal_components,
        columns=["Trục X (PCA 1)", "Trục Y (PCA 2)"],
        index=X.index,  # GIỮ chỉ mục để ghép tên phim / nhãn cụm
    )
    return df_pca
```

#### Nhiệm vụ 3 — Biểu đồ Radar (so sánh 2 bộ phim)

Hàm: `plot_radar_chart(df, movie1_name, movie2_name)` trong
`src/visualizer.py`

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def plot_radar_chart(df, movie1_name, movie2_name):
    # 1. Lọc 2 dòng dữ liệu của 2 phim
    movies_data = df[df["Title"].isin([movie1_name, movie2_name])]

    # 2. Chọn thuộc tính so sánh
    attributes = ["IMDb_Rating", "Votes",
                   "Budget_USD", "Revenue_USD", "Duration_min"]

    # 3. Min-Max Scaling về [0, 1] để 5 thuộc tính cùng thang
    scaler = MinMaxScaler()
    movies_scaled = scaler.fit_transform(movies_data[attributes])

    # 4. Góc các đỉnh radar + đóng vòng tròn
    angles = np.linspace(0, 2 * np.pi,
                         len(attributes), endpoint=False).tolist()
    angles += angles[:1]

    # 5. Vẽ
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    for i, row in enumerate(movies_scaled):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2,
                label=movies_data.iloc[i]["Title"])
        ax.fill(angles, values, alpha=0.25)

    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attributes)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    return fig
```

> ⚠️ Hàm này **bắt buộc** file CSV phải có cột `Votes` và
> `Duration_min` (xem mục 3.1).

---

### 📌 PHẦN IV — Dự đoán doanh thu (Linear Regression / Random Forest)

**File:** `src/models.py`

Hàm: `train_revenue_prediction_model(df)`

1. Mục tiêu dự đoán: `Revenue_USD`; đặc trưng gợi ý: `Budget_USD`,
   `IMDb_Rating`, `Votes`, `Duration_min` (không dùng `Title`/`Genre`
   thô nếu chưa mã hóa).
2. Bắt buộc **chia train/test** (`train_test_split`, `random_state=42`)
   rồi mới `fit` — tránh data leakage.
3. Dùng `Pipeline` + `ColumnTransformer` (điền khuyết, chuẩn hóa,
   mã hóa thể loại) để mô hình dự đoán được trên web.
4. Chọn `LinearRegression` **hoặc** `RandomForestRegressor`.
5. Đánh giá MAE, RMSE, R² trên tập test; trả về `Pipeline` đã huấn luyện.

---

### 👑 NHÓM TRƯỞNG — Tổng hợp lên `app.py`

- Import hàm từ `src/` vào 4 phần của sidebar (đã cấu trúc sẵn).
- Thiết kế nút bấm, ô nhập, bảng hiển thị, `st.pyplot(fig)`.
- Lưu dữ liệu sạch vào `st.session_state` để 4 trang dùng chung.
- Sau khi test thành công, các tổ **báo lại cho Nhóm trưởng** để
  import vào `app.py` — thành viên không tự sửa file này.

---

## 6. Checklist nghiệm thu cho từng tổ

- [ ] Code chạy được trên máy cá nhân (hoặc Google Colab).
- [ ] Tên hàm và tham số **không đổi** so với mục 4.
- [ ] Không còn chữ `pass` trong hàm phụ trách.
- [ ] Hàm trả về đúng kiểu dữ liệu đã quy ước.
- [ ] `calculate_basic_stats` tạo được `data/results2.csv`.
- [ ] Biểu đồ trả về `fig`, không gọi `plt.show()`.
- [ ] Không sửa `app.py`, `requirements.txt` (trừ khi báo Nhóm trưởng).
- [ ] Báo lại kết quả test cho Nhóm trưởng để tích hợp giao diện.

---

## 7. Ghi chú kỹ thuật

- **Khả năng tái lập:** mọi mô hình/PCA dùng `random_state=42`.
- **Chuẩn hóa dữ liệu:** K-means và PCA **bắt buộc** dùng
  `StandardScaler`; radar chart dùng `MinMaxScaler`.
- **Đóng hình:** sau `st.pyplot(fig)` nên `plt.close(fig)` để tránh
  rò rỉ bộ nhớ khi chạy lại app.
- **Bộ nhớ session:** mô hình huấn luyện nặng nên lưu vào
  `st.session_state`, đừng huấn luyện lại mỗi lần tương tác.
- **Lỗi phiên bản sklearn:** `python -m pip install -r requirements.txt`.

---

## 8. Cần thống nhất với Nhóm trưởng (chưa chốt)

1. **Cột dữ liệu ngoài bảng chuẩn:** radar chart cần `Votes`,
   `Duration_min` — đã thêm vào mục 3.1; Tổ 1 lưu ý khi thu thập thật.
2. **Chữ ký `run_pca_reduction(df)`:** giữ nguyên `df`, hàm tự scale
   bên trong (khác gợi ý nhận `X_scaled` — đã giải thích ở mục 5).
3. **Phần IV:** chưa có hướng dẫn chi tiết từ Nhóm trưởng; mục 5 chỉ
   là yêu cầu tối thiểu, chờ bổ sung.
