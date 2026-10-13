# 🎬 Movie Data Analysis & Prediction Web App (IMDb/TMDB)

Dự án phân tích dữ liệu phòng vé & dự đoán doanh thu phim bằng
**Streamlit + Pandas + Scikit-learn**.

> 📌 Tài liệu thi công cho các tổ. **Đọc kỹ mục 1 (trạng thái) và mục 2 (lỗi
> cần sửa) trước khi code** để không làm trùng hoặc đụng nhầm phần đã xong.

---

## 1. 📊 Trạng thái dự án

| Phần | File | Hàm / Script | Trạng thái | Ghi chú |
| --- | --- | --- | --- | --- |
| I | `src/collect_data.py` | `tai_file()`, `main()` | ✅ Xong | Tải TMDB 5000 (~46 MB) về `data/collected/` |
| I | `src/data_clean.py` | `clean_genre()`, `clean_currency()` | ✅ Xong | Xuất `data/cleaned/results.csv` (4.803 phim) |
| I | `src/data_pipeline.py` | `load_and_clean_data()` | ⬜ **Chưa ai nhận** | Đang là `pass` — xem mục 2.4 |
| II | `src/statistics_eda.py` | `calculate_basic_stats()` | 🔴 **Có lỗi** | 3 lỗi chính tả → crash. Xem mục 2.1 |
| II | `src/statistics_eda.py` | `get_top_bottom_movies()` | 🔴 **Có lỗi** | `Top_n` chưa định nghĩa. Xem mục 2.1 |
| II | `src/visualizer.py` | `plot_histogram()` | ⬜ **Chưa ai nhận** | Đang là `pass` — xem mục 2.2 |
| III | `src/models.py` | `run_kmeans_clustering()` | ✅ Xong | Test: K=3 → 1.540 / 1.367 / 322 phim |
| III | `src/models.py` | `run_pca_reduction()` | ✅ Xong | Giải thích 57,4% + 33,8% phương sai |
| III | `src/visualizer.py` | `plot_radar_chart()` | ✅ Xong | Đã test với dữ liệu thật |
| IV | `src/models.py` | `train_revenue_prediction_model()` | ✅ Xong | R² = 0,492 · MAE ≈ 73,6 triệu USD |
| UI | `app.py` | Phần III & Phần IV | ✅ Đã nối | Phần I & II đang là `st.info` chờ tổ |

**Tiến độ tổng thể: 8/10 hàm đã có code, 3 hàm đang lỗi hoặc chưa làm.**

---

## 2. 🐛 Danh sách lỗi & việc tồn đọng

### 2.1. 🔴 `src/statistics_eda.py` — 3 lỗi chính tả (Tổ 2 sửa ngay)

Đã chạy thử, **cả 2 hàm đều crash** ngay lập tức:

| Dòng | Code đang có | Phải sửa thành | Lỗi khi chạy |
| --- | --- | --- | --- |
| 35 | `pd.to_numeric(..., erors = "coerce")` | `errors="coerce"` | `TypeError: unexpected keyword argument 'erors'` |
| 41 | `numeric.syd(ddof = 1)` | `numeric.std(ddof=1)` | `AttributeError: no attribute 'syd'` |
| 76 | `data.nsmallest(Top_n, column)` | `data.nsmallest(top_n, column)` | `NameError: name 'Top_n' is not defined` |

Lỗi dòng 41 chưa lộ ra vì code chết sớm hơn ở dòng 35 — **sửa cả 3 chỗ rồi
test lại**, đừng sửa 1 chỗ rồi báo xong.

Lỗi nhỏ khác (PEP 8): dòng 8–9 thiếu dòng trống trước `def` (chuẩn là 2 dòng
trống giữa hằng số và hàm); dòng 27–28 và 35 có khoảng trắng thừa quanh dấu `=`.

### 2.2. ⬜ `plot_histogram(df, column)` — chưa ai nhận

Vẫn là `pass`, nên **Phần II chưa thể nối vào web**. Code tham khảo ở mục 7.3.
Ai nhận việc này thì nhớ báo Nhóm trưởng.

### 2.3. ⚠️ Thiếu cột `Votes` trong dữ liệu thật

`data/cleaned/results.csv` **không có cột `Votes`** (TMDB gốc là `vote_count`
nhưng `collect_data.py` không lấy). Hệ quả:

- `plot_radar_chart()` chỉ so sánh được **4 thuộc tính** (Rating, Budget,
  Revenue, Duration) thay vì 5 — vẫn chạy, nhưng báo cáo phải nói rõ.
- `train_revenue_prediction_model()` tự bỏ qua `Votes` nên không ảnh hưởng.

→ Muốn đủ 5 thuộc tính thì Tổ 1 thêm `vote_count` vào `collect_data.py` rồi
chạy lại 2 bước pipeline (mục 3).

### 2.4. ⬜ `load_and_clean_data(raw_file_path)` — chưa ai nhận

Vẫn là `pass`. Hiện logic làm sạch đang nằm ở `data_clean.py` (dạng script
chạy từ trên xuống, không có hàm, không nhận tham số). Việc của Tổ 1 là
**đóng gói logic đó thành hàm** `load_and_clean_data(raw_file_path)` để
`app.py` gọi được — xem mục 7.1.

### 2.5. 🧹 Vệ sinh repo

- `.idea/` (config PyCharm) bị commit nhầm → đã thêm `.gitignore` và gỡ khỏi git.
- `data/phim.csv` đã xóa trong máy nhưng vẫn còn trong git.
- **Repo không có `src/BTL.py`** — script lấy Actors/Director đó đang nằm ở
  máy cá nhân. `collect_data.py` đã làm đủ việc đó nên không cần commit thêm.

---

## 3. 🔄 Pipeline dữ liệu (chạy đúng thứ tự)

```powershell
Set-Location "d:\PRJ\phimro-database"

# Bước 1 — Tải dataset TMDB 5000 (~46 MB, mất 1-2 phút)
python src\collect_data.py
#   -> data/collected/tmdb_5000_movies.csv
#   -> data/collected/tmdb_5000_credits.csv
#   -> data/collected/resultsRaw_completed.csv

# Bước 2 — Làm sạch, đổi tên cột, lọc trùng
python src\data_clean.py
#   -> data/cleaned/results.csv   (4.803 phim)

# Bước 3 — Chạy web
python -m streamlit run app.py
```

> ⚠️ **Cả 2 lệnh phải chạy từ thư mục gốc** `d:\PRJ\phimro-database` vì
> `data_clean.py` dùng đường dẫn tương đối.
>
> ⚠️ `data/` **không được commit** (xem `.gitignore`). Ai mới clone repo phải
> tự chạy bước 1 và 2 để có dữ liệu.

---

## 4. 📁 Cấu trúc dự án

```text
phimro-database/
├── data/
│   ├── .gitkeep
│   ├── cleaned/
│   │   └── results.csv                # Dữ liệu sạch — app.py đọc file này
│   └── collected/                     # Dataset thô (không commit)
│       ├── tmdb_5000_movies.csv
│       ├── tmdb_5000_credits.csv
│       └── resultsRaw_completed.csv
├── src/
│   ├── collect_data.py                # Bước 1: thu thập (Tổ 1)
│   ├── data_clean.py                  # Bước 2: làm sạch (Tổ 1)
│   ├── data_pipeline.py               # Phần I    - hợp đồng load_and_clean_data
│   ├── statistics_eda.py              # Phần II   - thống kê mô tả (Tổ 2)
│   ├── models.py                      # Phần III & IV - học máy (Tổ 3)
│   └── visualizer.py                  # Phần II & III - biểu đồ
├── app.py                             # Giao diện Streamlit (Nhóm trưởng)
├── requirements.txt
└── README.md
```

---

## 5. 📋 Quy ước dữ liệu

### 5.1. Cột thực tế trong `data/cleaned/results.csv`

4.803 dòng × 9 cột, tên cột **đã rút gọn**:

| Cột | Kiểu | Ví dụ | Ghi chú |
| --- | --- | --- | --- |
| `Title` | str | `Avatar` | Khóa tìm kiếm phim |
| `Year` | str | `2009` | ⚠️ Là chuỗi do bước `fillna('N/a')` |
| `Genre` | str | `Action, Adventure, Fantasy` | Đã bóc tách từ JSON |
| `Duration` | str/num | `162` | ⚠️ Có thể là `"N/a"` |
| `Rating` | float | `7.9` | Điểm TMDB 0–10 (thay cho IMDb) |
| `Budget` | str/num | `237000000` | ⚠️ Có thể là `"N/a"` |
| `Revenue` | str/num | `2787965087` | ⚠️ Có thể là `"N/a"` |
| `Director` | str | `James Cameron` | |
| `Cast` | str | `Sam Worthington, Zoe Saldana, ...` | Top 10 diễn viên |
| ~~`Votes`~~ | — | — | ❌ **Không có** — xem mục 2.3 |

### 5.2. Bảng đổi tên 2 chiều (quan trọng)

Code trong `models.py` / `visualizer.py` **chấp nhận cả 2 bộ tên** nhờ bảng
ánh xạ, nên dù Tổ 1 đổi tên hay không thì không crash:

| Tên chuẩn (hợp đồng) | Tên rút gọn (dữ liệu thật) |
| --- | --- |
| `IMDb_Rating` | `Rating` |
| `Budget_USD` | `Budget` |
| `Revenue_USD` | `Revenue` |
| `Duration_min` | `Duration` |
| `Votes` | — *(chưa có)* |

**Nguyên tắc:** khi viết code mới, dùng **tên chuẩn** và tra bảng này để
fallback sang tên rút gọn (mẫu code ở mục 7.5). Không hard-code 1 tên duy nhất.

### 5.3. ⚠️ Lưu ý kiểu dữ liệu

`data_clean.py` bước 8 dùng `df.fillna('N/a')` nên **`Year`, `Duration`,
`Budget`, `Revenue` đang là kiểu `object`**, không phải số. Ai đọc các cột
này phải tự ép kiểu:

```python
pd.to_numeric(df["Budget"], errors="coerce")   # "N/a" -> NaN
```

Các hàm trong `models.py` đã tự làm việc này nên không sao — nhưng nếu ai đọc
trực tiếp thì sẽ sai số liệu. **Đây là bug nên Tổ 1 cân nhắc bỏ
`fillna('N/a')`** (để `NaN` thật) và xử lý hiển thị ở tầng UI.

---

## 6. 📜 Hợp đồng hàm — tra cứu nhanh

| Hàm | File | Nhận vào | Trả về |
| --- | --- | --- | --- |
| `load_and_clean_data(raw_file_path)` | `data_pipeline.py` | đường dẫn CSV thô | `DataFrame` đã làm sạch |
| `calculate_basic_stats(df)` | `statistics_eda.py` | DataFrame | `DataFrame` có cột `Mean`, `Median`, `Std`; index tên `column` |
| `get_top_bottom_movies(df, column, top_n=3)` | `statistics_eda.py` | DataFrame, tên cột, số dòng | `(top_movies, bottom_movies)` — 2 DataFrame cột `Title` + `column` |
| `plot_histogram(df, column)` | `visualizer.py` | DataFrame, tên cột | `matplotlib.figure.Figure` |
| `plot_radar_chart(df, movie1_name, movie2_name)` | `visualizer.py` | DataFrame, 2 tên phim | `matplotlib.figure.Figure` |
| `run_kmeans_clustering(df, n_clusters)` | `models.py` | DataFrame, số cụm | DataFrame 3 đặc trưng + cột `Cluster`, giữ index gốc |
| `run_pca_reduction(df)` | `models.py` | DataFrame | DataFrame `Trục X (PCA 1)`, `Trục Y (PCA 2)`; `.attrs["explained_variance_ratio"]` |
| `train_revenue_prediction_model(df)` | `models.py` | DataFrame | `Pipeline` sklearn; `.metrics_`, `.feature_columns_` |

**Quy tắc chung khi trả về hình:** trả về đối tượng `fig`, **không** gọi
`plt.show()`; `app.py` sẽ dùng `st.pyplot(fig)` rồi `plt.close(fig)`.

**Quy tắc chung khi báo lỗi:** ném `TypeError` nếu `df` sai kiểu,
`ValueError` kèm **thông báo tiếng Việt rõ ràng** nếu thiếu cột hoặc dữ liệu
không đủ — `app.py` bắt `ValueError` rồi hiển thị bằng `st.error()`.


---

## 7. 🧰 Code mẫu & ghi chú kỹ thuật

### 7.1. `load_and_clean_data(raw_file_path)` — mẫu đóng gói (Tổ 1)

Lấy logic của `data_clean.py` bỏ vào hàm, nhận đường dẫn, **trả về DataFrame**
(chứ không tự ghi file — để `app.py` quyết định):

```python
import json
import pathlib
import re

import numpy as np
import pandas as pd


def _extract_genre(val):
    """Bóc tên thể loại ra khỏi chuỗi JSON trong cột Genre."""
    if pd.isna(val):
        return np.nan
    try:
        s = str(val).strip().strip('"').replace('""', '"')
        names = [item["name"] for item in json.loads(s) if "name" in item]
        return ", ".join(names) if names else np.nan
    except (ValueError, TypeError):
        names = re.findall(r'"name"\s*:\s*"([^"]+)"', str(val))
        return ", ".join(names) if names else np.nan


def load_and_clean_data(raw_file_path):
    """Đọc dữ liệu thô và trả về DataFrame đã được làm sạch."""
    path = pathlib.Path(raw_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu: {path}")

    # keep_default_na=False: chỉ ta tự quyết định chuỗi nào là giá trị thiếu
    df = pd.read_csv(path, keep_default_na=False, na_values=["N/a", "", "nan"])

    df = df.rename(columns={
        "Release_Year": "Year",
        "IMDb_Rating": "Rating",
        "Budget_USD": "Budget",
        "Revenue_USD": "Revenue",
        "Duration_min": "Duration",
        "Actors": "Cast",
    })

    required = ["Title", "Genre", "Rating", "Budget", "Revenue"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {', '.join(missing)}.")

    if "Genre" in df.columns:
        df["Genre"] = df["Genre"].apply(_extract_genre)
    for col in ("Year", "Duration", "Rating", "Budget", "Revenue"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.drop_duplicates(subset=["Title", "Year"])
    df = df.sort_values(["Title", "Year"], ascending=[True, False])

    # Giữ NaN thay vì 'N/a' để các bước thống kê/học máy tính đúng
    return df.reset_index(drop=True)
```

> ⚠️ Lưu ý: mẫu trên **không** `fillna('N/a')` — xem mục 5.3 để biết vì sao.

### 7.2. `calculate_basic_stats(df)` — mẫu kèm nhóm theo Genre (Tổ 2)

```python
def calculate_basic_stats(df):
    cols = ["Duration", "Rating", "Budget", "Revenue"]

    # 1. Thống kê toàn bộ dữ liệu
    numeric = df[cols].apply(pd.to_numeric, errors="coerce")
    overall = pd.DataFrame({
        "Mean": numeric.mean(),
        "Median": numeric.median(),
        "Std": numeric.std(ddof=1),
    })
    overall.index.name = "column"

    # 2. Thống kê theo từng thể loại (phim đa thể loại thì tách bằng ", ")
    genres = (
        df["Genre"].fillna("Unknown").str.split(",")
        .explode().str.strip()
    )
    by_genre = numeric.groupby(genres).agg(["mean", "median", "std"])

    return overall, by_genre
```

Xuất CSV: `overall.to_csv("data/results2.csv", index=True)`.
Genre dạng `"Action, Drama"` nên phải `.explode()` trước khi `groupby`,
nếu không mỗi cụm chữ sẽ thành 1 nhóm riêng.


### 7.3. `plot_histogram(df, column)` — mẫu (ai nhận việc này)

```python
import matplotlib.pyplot as plt


def plot_histogram(df, column):
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        raise ValueError(f"Cột '{column}' không có giá trị số hợp lệ.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=20, color="skyblue", edgecolor="black")
    ax.set_title(f"Phân bố của {column}")
    ax.set_xlabel(column)
    ax.set_ylabel("Số lượng phim")
    ax.grid(axis="y", alpha=0.3)
    return fig  # Streamlit tự st.pyplot(fig); KHÔNG gọi plt.show()
```

### 7.4. Elbow Method — chọn số cụm cho K-means

Bắt buộc có trong báo cáo để biện luận vì sao chọn K:

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from src.models import _prepare_cluster_features

df = pd.read_csv("data/cleaned/results.csv")
X = _prepare_cluster_features(df)          # đã ép số + dropna
X_scaled = StandardScaler().fit_transform(X)

scores = {}
for k in range(1, 11):
    scores[k] = KMeans(k, random_state=42, n_init=10).fit(X_scaled).inertia_

plt.plot(list(scores), list(scores.values()), "o-")
plt.xlabel("Số cụm K")
plt.ylabel("Inertia (tổng bình phương khoảng cách đến tâm cụm)")
plt.title("Elbow Method")
plt.show()   # chọn K tại điểm gãy khúc — đồ thị giảm chậm lại đột ngột
```

### 7.5. Mẫu ánh xạ tên cột 2 chiều (dùng cho mọi code mới)

```python
_COLUMN_ALIASES = {
    "IMDb_Rating": ("IMDb_Rating", "Rating"),
    "Budget_USD": ("Budget_USD", "Budget"),
    "Revenue_USD": ("Revenue_USD", "Revenue"),
    "Duration_min": ("Duration_min", "Duration"),
}


def resolve_columns(df, aliases):
    """Tìm cột thật trong df theo danh sách tên chấp nhận được.

    Trả về (dict tên chuẩn -> tên cột thật, danh sách tên chuẩn bị thiếu).
    """
    found, missing = {}, []
    for canonical, candidates in aliases.items():
        hit = next((c for c in candidates if c in df.columns), None)
        if hit is None:
            missing.append(canonical)
        else:
            found[canonical] = hit
    return found, missing
```

---

## 8. ⚠️ Nguyên tắc bắt buộc khi code

1. **Giữ nguyên tên hàm và tham số** trong bảng mục 6. Cần tham số mới thì
   phải có giá trị mặc định **và báo Nhóm trưởng trước**.
2. **Không tự sửa `app.py`** hay `requirements.txt`. Code xong thì báo để
   Nhóm trưởng nối vào giao diện.
3. **Thư viện chỉ dùng:** `pandas`, `numpy`, `matplotlib.pyplot`, `sklearn`
   (và `requests` + `beautifulsoup4` cho phần thu thập).
4. **Chạy mọi lệnh test từ thư mục gốc** `d:\PRJ\phimro-database`.
5. **Không commit file CSV** trong `data/` (đã chặn ở `.gitignore`).
6. **Giữ docstring tiếng Việt**; ghi chú của mình thêm vào cuối docstring.
7. **Không sửa `df` đầu vào** — luôn làm việc trên bản sao (`.copy()`).
8. **Không điền giá trị suy đoán vào cột mục tiêu** (`Revenue`) khi thiếu.

---

## 9. ✅ Checklist nghiệm thu từng tổ

- [ ] Code chạy được, **không còn chữ `pass`** trong hàm phụ trách.
- [ ] Tên hàm và tham số **không đổi** so với bảng mục 6.
- [ ] Đã test bằng dữ liệu thật `data/cleaned/results.csv`, không chỉ mock.
- [ ] Đã test cả trường hợp **lỗi** (thiếu cột, cột toàn `"N/a"`) → phải ra
      `ValueError` có thông báo tiếng Việt.
- [ ] Hàm trả về đúng kiểu dữ liệu đã quy ước.
- [ ] `calculate_basic_stats` tạo được `data/results2.csv`.
- [ ] Hàm vẽ trả về `fig`, **không** gọi `plt.show()`.
- [ ] Không sửa `app.py`, `requirements.txt` (trừ khi báo Nhóm trưởng).
- [ ] **Báo kết quả test cho Nhóm trưởng** để tích hợp giao diện.

---

## 10. 🔧 Ghi chú kỹ thuật

- **Khả năng tái lập:** mọi mô hình/PCA/K-means dùng `random_state=42`;
  `KMeans(n_init=10)` để không phụ thuộc mặc định đổi theo phiên bản sklearn.
- **Chuẩn hóa:** K-means và PCA **bắt buộc** `StandardScaler`; radar chart
  dùng `MinMaxScaler` học trên **toàn bộ** dữ liệu chứ không chỉ 2 phim.
- **Chống data leakage:** chia train/test **trước** khi `fit` bước tiền xử lý.
- **Đóng hình:** sau `st.pyplot(fig)` thì `plt.close(fig)`.
- **Bộ nhớ session:** mô hình huấn luyện nặng thì lưu vào `st.session_state`,
  đừng train lại mỗi lần tương tác.
- **Lỗi phiên bản sklearn:** `python -m pip install -r requirements.txt`.

---

## 11. 📌 Việc cần Nhóm trưởng quyết

1. **Thêm cột `Votes`** vào `collect_data.py` (`vote_count`) và chạy lại
   pipeline để radar chart đủ 5 thuộc tính? (mục 2.3)
2. **Bỏ `fillna('N/a')`** trong `data_clean.py` để dữ liệu là số thật? (mục 5.3)
3. **Ai nhận** `plot_histogram()` và `load_and_clean_data()`? (mục 2.2, 2.4)
4. **Phân công viết báo cáo** (Elbow Method, giải thích R² = 0,49, so sánh
   LinearRegression vs RandomForest) — Tổ 3 hay cả lớp?


### 7.6. Nối hàm vào `app.py`

Chỉ **Nhóm trưởng** sửa `app.py`. Mẫu chuẩn cho Phần II:

```python
elif section == "Statistics & Distribution (Part II)":
    df = load_cleaned_data()
    if df is None:
        st.error("Chưa có dữ liệu sạch — hãy chạy bước làm sạch trước.")
    else:
        column = st.selectbox(
            "Chọn chỉ số", ["Rating", "Budget", "Revenue", "Duration"]
        )
        stats = calculate_basic_stats(df)
        st.dataframe(stats, use_container_width=True)

        top, bottom = get_top_bottom_movies(df, column)
        st.subheader(f"Top 3 phim cao nhất theo {column}")
        st.dataframe(top, use_container_width=True)
        st.subheader(f"Top 3 phim thấp nhất theo {column}")
        st.dataframe(bottom, use_container_width=True)

        fig = plot_histogram(df, column)
        st.pyplot(fig)
        plt.close(fig)   # tránh rò rỉ bộ nhớ khi Streamlit chạy lại
```


