# 🎬 Movie Intelligence: Phân Tích Dữ Liệu Điện Ảnh & Dự Đoán Doanh Thu Phòng Vé (IMDb/TMDB)

Ứng dụng web trực quan hóa dữ liệu và dự báo doanh thu phòng vé được xây dựng trên nền tảng **Python**, **Streamlit** và **Scikit-Learn**. 

Hệ thống kết hợp kỹ thuật **Khai phá dữ liệu (Data Mining)**, **Thống kê mô tả (EDA)** và các thuật toán **Học máy (Machine Learning)** nhằm cung cấp góc nhìn toàn diện về thị trường điện ảnh thế giới.

---

## 📌 1. Mục đích và Ý tưởng của Dự án

Trong ngành công nghiệp điện ảnh toàn cầu, việc sản xuất một bộ phim đòi hỏi nguồn vốn đầu tư khổng lồ (hàng chục đến hàng trăm triệu USD). Tuy nhiên, mức độ thành công tại phòng vé cũng như phản ứng của khán giả và giới phê bình phụ thuộc vào sự đan xen phức tạp giữa nhiều yếu tố: **kinh phí sản xuất (Budget)**, **thể loại (Genre)**, **thời lượng (Duration)**, **sức hút diễn viên/đạo diễn (Cast/Director)** và **chất lượng nội dung (Rating & Votes)**.

🎯 **Mục tiêu cốt lõi của Web App:**
1. **Dân chủ hóa và trực quan hóa dữ liệu điện ảnh:** Chuyển đổi các tập dữ liệu thô, phức tạp từ IMDb và TMDB thành các biểu đồ trực quan, bảng biểu phân tích tương tác dễ hiểu cho người xem.
2. **Khám phá quy luật thị trường (Exploratory Data Analysis):** Giúp người xem nắm bắt xu hướng doanh thu, kinh phí, điểm đánh giá theo từng thể loại và thời kỳ phát hành.
3. **Phân khúc thị trường phim (Clustering & Dimensionality Reduction):** Nhóm các bộ phim có đặc tính tương đồng (bom tấn sinh lời lớn, phim nghệ thuật kinh phí thấp điểm cao, phim thua lỗ...) thông qua mô hình phân cụm K-Means và bản đồ tọa độ PCA 2 chiều.
4. **Đối đầu và so sánh tác phẩm (Radar Comparative Analysis):** Cung cấp công cụ so sánh đa chiều giữa hai tác phẩm điện ảnh bất kỳ để nhận diện điểm mạnh, điểm yếu trên cùng một hệ quy chiếu.
5. **Dự báo tiềm năng thương mại (Revenue Prediction):** Ứng dụng mô hình học máy hồi quy (Regression) để dự đoán doanh thu phòng vé dựa trên các tham số đầu vào được thiết lập trước khi phát hành, hỗ trợ việc đánh giá tiềm năng thị trường.

👥 **Đối tượng phục vụ:**
- **Khán giả và người yêu điện ảnh:** Tra cứu thông tin, khám phá xu hướng các dòng phim yêu thích và so sánh các tác phẩm kinh điển.
- **Nhà nghiên cứu / Giới phê bình:** Khai thác dữ liệu thống kê, quan sát tương quan giữa đánh giá chuyên môn và sức tiêu thụ vé.
- **Nhà đầu tư / Nhà sản xuất phim:** Tham khảo dữ liệu lịch sử để đánh giá tương quan giữa kinh phí đầu tư và tiềm năng doanh thu phòng vé.

---

## 🌟 2. Các Phân Hệ & Tính Năng Chính Của Web App

Giao diện ứng dụng được tổ chức khoa học thành **4 phân hệ chính** trên thanh điều hướng Streamlit:

```
                            🌐 GIAO DIỆN WEB STREAMLIT
                                        │
    ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
    ▼                   ▼                               ▼                   ▼
[Phần I: Pipeline]  [Phần II: Thống kê & EDA]    [Phần III: Phân cụm]   [Phần IV: Dự đoán]
- Làm sạch dữ liệu  - Thống kê All & Theo Thể loại- Phân cụm K-Means    - Mô hình Hồi quy
- Xử lý thiếu/trùng - Top/Bottom phim             - Bản đồ PCA 2D       - Nhập thông số phim
- Chuẩn hóa 9 cột   - Phân bố Histogram           - Biểu đồ Radar       - Dự báo doanh thu
```

---

### 🔹 Phần I: Tổng quan & Tiền xử lý Dữ liệu (Overview & Data Pipeline)
*Module nền tảng đảm bảo chất lượng và độ tin cậy của toàn bộ luồng phân tích.*
- **Tải và khám phá dữ liệu:** Xem trước bảng dữ liệu phim thô thu thập từ IMDb/TMDB (`resultsRaw_completed.csv` / `results.csv`).
- **Chuẩn hóa 9 thuộc tính tiêu chuẩn:** `Title` (Tên phim), `Year` (Năm phát hành), `Genre` (Thể loại), `Duration` (Thời lượng), `Rating` (Điểm IMDb), `Budget` (Kinh phí), `Revenue` (Doanh thu), `Director` (Đạo diễn), `Cast` (Dàn diễn viên).
- **Làm sạch sâu:** Xử lý chuỗi JSON thể loại, chuẩn hóa ký hiệu tiền tệ (`$`, dấu phẩy), khử giá trị khuyết thiếu (`NaN`, `N/a`) và loại bỏ bản ghi trùng lặp (`drop_duplicates`).

### 🔹 Phần II: Thống kê Mô tả & Phân tích Thị trường (Statistics & Distribution)
*Cung cấp bức tranh toàn cảnh về hiệu quả tài chính và chất lượng nghệ thuật.*
- **Chỉ số thống kê đa chiều:** Tự động tính toán các đại lượng **Mean (Trung bình)**, **Median (Trung vị)**, **Std (Độ lệch chuẩn)** của Doanh thu, Kinh phí và Điểm đánh giá cho toàn bộ tập dữ liệu (`All`) cũng như phân nhóm theo từng thể loại (`Genre`).
- **Bảng xếp hạng Top & Bottom:** Tìm kiếm nhanh top các bộ phim có điểm số cao nhất/thấp nhất hoặc doanh thu ấn tượng nhất để phân tích yếu tố thành công hay thất bại.
- **Trực quan hóa phân phối (Histogram):** Vẽ biểu đồ tần suất phân bố của điểm đánh giá, ngân sách và doanh thu để người xem nhận diện độ lệch (skewness) của thị trường.

### 🔹 Phần III: Phân cụm Thị trường & So sánh Đa Chiều (Clustering & Radar Comparison)
*Khai phá cấu trúc tiềm ẩn và công cụ so sánh trực quan sâu sắc.*
- **Phân cụm K-Means:** Tự động chuẩn hóa dữ liệu với `StandardScaler` và gom nhóm các tác phẩm điện ảnh vào các phân khúc đặc trưng (ví dụ: Bom tấn đại chúng, Phim kinh phí trung bình sinh lời, Phim độc lập nghệ thuật).
- **Chiếu bản đồ 2D (PCA):** Sử dụng thuật toán Giảm chiều dữ liệu PCA (Principal Component Analysis) để biểu diễn các cụm phim trên mặt phẳng 2D, giúp người dùng dễ dàng quan sát mật độ và ranh giới giữa các nhóm phim.
- **Biểu đồ Radar (Spider Chart):** Công cụ so sánh đối đầu giữa hai bộ phim bất kỳ dựa trên các thang đo đã được chuẩn hóa `MinMaxScaler` (Rating, Budget, Revenue, Duration, Votes), làm nổi bật thế mạnh tương đối của từng tác phẩm.

### 🔹 Phần IV: Mô hình Dự đoán Doanh thu Phòng vé (Box Office Revenue Prediction)
*Ứng dụng trí tuệ nhân tạo để lượng hóa tiềm năng thương mại.*
- **Huấn luyện mô hình Machine Learning:** Tích hợp `LinearRegression` hoặc `RandomForestRegressor` được đóng gói trong quy trình chuẩn `Pipeline` và `ColumnTransformer` của Scikit-Learn.
- **Đánh giá khách quan:** Tách bạch tập kiểm tra độc lập (`train_test_split`), báo cáo đầy đủ các chỉ số đo lường sai số: **MAE** (Sai số tuyệt đối trung bình), **RMSE** (Căn bậc hai sai số toàn phương trung bình) và hệ số xác định **$R^2$ Score**.
- **Mô phỏng doanh thu tương tác:** Cho phép người dùng nhập các thông số dự kiến của một dự án phim mới (kinh phí dự kiến, thể loại, thời lượng, điểm đánh giá kỳ vọng...) để mô hình ước tính doanh thu phòng vé tiềm năng ngay trên trình duyệt.

---

## 🛠️ 3. Công Nghệ & Thư Viện Sử Dụng

| Hạng mục | Công nghệ / Thư viện | Mục đích sử dụng |
|---|---|---|
| **Giao diện Web** | Streamlit | Xây dựng Dashboard web tương tác trực quan, thời gian thực |
| **Xử lý dữ liệu** | Pandas, NumPy | Đọc, lọc, chuyển đổi kiểu, biến đổi ma trận và làm sạch dữ liệu |
| **Học máy & Khai phá** | Scikit-Learn | K-Means, PCA, StandardScaler, MinMaxScaler, LinearRegression / RandomForest |
| **Trực quan hóa đồ họa** | Matplotlib, Seaborn | Vẽ biểu đồ Histogram phân phối, bản đồ cụm 2D và biểu đồ Radar cực đa giác |
| **Thu thập dữ liệu** | BeautifulSoup4, Requests | Hỗ trợ cào dữ liệu phim từ các nguồn mở (IMDb/TMDB) |

---

## 📂 4. Cấu Trúc Thư Mục Dự Án

```text
BTL Python/
├── data/
│   ├── phim.csv                 # Tập dữ liệu phim đã làm sạch chuẩn bị sẵn
│   ├── results.csv              # Dữ liệu bảng phim chuẩn dùng cho phân tích
│   └── results2.csv             # Kết quả thống kê mô tả xuất ra từ Phần II
├── src/
│   ├── data_clean.py            # Script tiền xử lý từ dữ liệu thô sang dữ liệu chuẩn
│   ├── data_pipeline.py         # Module Phần I: Chuẩn hóa & kiểm tra dữ liệu đầu vào
│   ├── statistics_eda.py        # Module Phần II: Thống kê Mean/Median/Std & Top phim
│   ├── models.py                # Module Phần III & IV: K-Means, PCA & Mô hình Hồi quy
│   └── visualizer.py            # Module Phần II & III: Đồ họa Histogram & Radar Chart
├── app.py                       # Điểm khởi chạy ứng dụng web Streamlit
├── requirements.txt             # Danh sách gói thư viện Python cần thiết
└── README.md                    # Tài liệu giới thiệu & hướng dẫn dự án
```

---

## 🚀 5. Hướng Dẫn Cài Đặt & Khởi Chạy

### Yêu cầu tiên quyết:
- Đã cài đặt **Python 3.10** trở lên trên hệ điều hành (Windows / macOS / Linux).

### Các bước thực hiện:

1. **Mở terminal/Powershell tại thư mục dự án:**
   ```powershell
   cd "c:\BTL Python\BTL Python"
   ```

2. **Khởi tạo môi trường ảo (khuyến nghị):**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   *(Trên Linux/macOS: `source .venv/bin/activate`)*

3. **Cài đặt các thư viện cần thiết:**
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. **Khởi chạy ứng dụng Web:**
   ```powershell
   python -m streamlit run app.py
   ```

5. **Truy cập ứng dụng:**
   - Mở trình duyệt web và truy cập địa chỉ: `http://localhost:8501`
   - Sử dụng thanh menu bên trái (Sidebar) để chuyển đổi qua lại giữa 4 phân hệ tính năng.

---

## 📊 6. Cấu Trúc Bảng Dữ Liệu Chuẩn

Hệ thống sử dụng bảng dữ liệu phim với các trường thông tin chuẩn hóa:

| Tên cột | Kiểu dữ liệu | Mô tả | Ý nghĩa phân tích |
|---|---|---|---|
| `Title` | String | Tên chính thức của bộ phim | Khóa định danh để tra cứu, hiển thị và so sánh |
| `Year` | Integer | Năm công chiếu | Phân tích xu hướng phim theo thời gian |
| `Genre` | String | Danh sách thể loại (Action, Drama, ...) | Phân nhóm, thống kê thị hiếu theo dòng phim |
| `Duration` | Integer | Thời lượng phim (phút) | Đánh giá độ dài tác phẩm tác động đến người xem |
| `Rating` | Float | Điểm số đánh giá trên thang 10 | Đo lường chất lượng và phản hồi từ khán giả |
| `Budget` | Float / Int | Kinh phí sản xuất (USD) | Đầu vào tài chính, chi phí đầu tư ban đầu |
| `Revenue` | Float / Int | Doanh thu phòng vé toàn cầu (USD) | Thước đo thành công thương mại, biến mục tiêu dự đoán |
| `Director` | String | Đạo diễn bộ phim | Phân tích phong cách nghệ thuật và bảo chứng phòng vé |
| `Cast` | String | Các diễn viên chính tham gia | Yếu tố ngôi sao thu hút khán giả đến rạp |

---

## ⚙️ 7. Hợp Đồng Mô-Đun Kỹ Thuật (API Specifications)

Hệ thống được thiết kế theo nguyên lý tách biệt giao diện và tầng logic (Modular Architecture):

- `load_and_clean_data(raw_file_path)`: Đọc và trả về DataFrame đã làm sạch kiểu dữ liệu, loại bỏ giá trị khuyết thiếu và trùng lặp.
- `calculate_basic_stats(df)`: Tổng hợp các chỉ số Mean, Median, Std cho toàn bộ bảng và theo nhóm thể loại; xuất dữ liệu ra `data/results2.csv`.
- `get_top_bottom_movies(df, column, top_n=3)`: Trả về cặp `(top_movies, bottom_movies)` theo thuộc tính lựa chọn.
- `plot_histogram(df, column)`: Sinh đối tượng `Figure` biểu đồ phân phối tần suất của một thuộc tính số.
- `run_kmeans_clustering(df, n_clusters)`: Chuẩn hóa đặc trưng và phân cụm K-Means, bổ sung nhãn `Cluster` vào dữ liệu.
- `run_pca_reduction(df)`: Giảm không gian dữ liệu đa chiều về 2 chiều (PC1, PC2) để trực quan hóa cụm phim.
- `plot_radar_chart(df, movie1_name, movie2_name)`: Trực quan hóa so sánh đa chiều giữa hai bộ phim trên đồ thị tọa độ cực (Polar/Radar Chart).
- `train_revenue_prediction_model(df)`: Đào tạo và đánh giá mô hình Scikit-Learn Pipeline dự đoán doanh thu phòng vé.

---

## 🔮 8. Định Hướng Phát Triển Mở Rộng

- [ ] **Mở rộng dữ liệu:** Tích hợp API tự động đồng bộ thời gian thực từ The Movie Database (TMDb API).
- [ ] **Phân tích Cảm xúc (Sentiment Analysis):** Khai phá các đánh giá bằng văn bản của khán giả từ IMDb để lượng hóa chỉ số yêu thích nội dung.
- [ ] **Mô hình học máy nâng cao:** Bổ sung thuật toán Gradient Boosting (`XGBoost`, `LightGBM`) để gia tăng độ chính xác dự báo doanh thu.
- [ ] **Đề xuất phim thông minh (Recommender System):** Đề xuất các bộ phim tương đồng dựa trên khoảng cách đặc trưng nội dung (Content-based filtering).
