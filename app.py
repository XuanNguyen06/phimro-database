"""Giao diện Streamlit cho ứng dụng phân tích và dự đoán dữ liệu phim."""

import matplotlib

matplotlib.use("Agg")  # backend không cửa sổ, an toàn khi chạy trong Streamlit

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src.config import (
    COLUMN_LABELS,
    DEFAULT_DATA,
    MISSING_LABEL,
    MONEY_COLUMNS,
    NUMERIC_COLUMNS,
    RESULTS_FILE,
    STATS_FILE,
)
from src.data_pipeline import load_and_clean_data
from src.models import (
    evaluate_k_range,
    get_feature_importances,
    run_kmeans_clustering,
    run_pca_reduction,
    train_revenue_prediction_model,
)
from src.statistics_eda import (
    calculate_basic_stats,
    calculate_stats_by_genre,
    get_top_bottom_movies,
)
from src.visualizer import plot_histogram, plot_pca_clusters, plot_radar_chart

st.set_page_config(
    page_title="Movie Data Analysis & Prediction",
    page_icon="🎬",
    layout="wide",
)


# ----------------------------------------------------------------------------
# Hàm tiện ích
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Đang đọc và làm sạch dữ liệu...")
def load_default_data(path_str, _mtime):
    """Đọc + làm sạch tệp mặc định; _mtime giúp tự làm mới cache khi tệp đổi."""
    return load_and_clean_data(path_str)


def get_data():
    """Trả về (DataFrame sạch, tên nguồn). Ưu tiên tệp người dùng tải lên."""
    if "df" in st.session_state:
        return st.session_state["df"], st.session_state["source"]
    if not DEFAULT_DATA.is_file():
        st.error(f"Không tìm thấy dữ liệu mặc định: {DEFAULT_DATA}")
        st.stop()
    df = load_default_data(str(DEFAULT_DATA), DEFAULT_DATA.stat().st_mtime)
    return df, DEFAULT_DATA.name


def money_config(columns):
    """Cấu hình hiển thị tiền tệ cho st.dataframe."""
    return {
        c: st.column_config.NumberColumn(c, format="dollar")
        for c in columns if c in MONEY_COLUMNS
    }


def show_figure(fig):
    """Hiển thị hình Matplotlib rồi đóng để giải phóng bộ nhớ."""
    st.pyplot(fig)
    plt.close(fig)


@st.cache_resource(show_spinner="Đang huấn luyện mô hình RandomForest...")
def train_model_cached(df_model):
    return train_revenue_prediction_model(df_model)


@st.cache_data(show_spinner="Đang thử K từ 2 đến 8...")
def evaluate_k_cached(df_cluster):
    return evaluate_k_range(df_cluster, 2, 8)


# ----------------------------------------------------------------------------
# Khung trang
# ----------------------------------------------------------------------------
st.title("🎬 Phân tích dữ liệu và dự đoán doanh thu phim")
st.caption("Dự án phân tích dữ liệu IMDb/TMDB")

section = st.sidebar.radio(
    "Điều hướng",
    (
        "Overview & Data Cleaning (Part I)",
        "Statistics & Distribution (Part II)",
        "Clustering & Radar Chart (Part III)",
        "Revenue Prediction (Part IV)",
    ),
)

df, source_name = get_data()
st.sidebar.divider()
st.sidebar.markdown(f"**Nguồn dữ liệu:** `{source_name}`")
st.sidebar.markdown(f"**Số phim sau làm sạch:** {len(df):,}")


# ----------------------------------------------------------------------------
# PHẦN I
# ----------------------------------------------------------------------------
if section == "Overview & Data Cleaning (Part I)":
    st.header("Phần I: Tổng quan và làm sạch dữ liệu")

    with st.expander("📤 Dùng tệp dữ liệu khác (tùy chọn)"):
        st.write(
            "Mặc định ứng dụng dùng `data/phim.csv`. Bạn có thể tải lên tệp CSV thô "
            "(ví dụ `resultsRaw_completed.csv`) — hệ thống tự đổi tên cột, tách thể loại "
            "JSON và chuẩn hóa tiền tệ."
        )
        uploaded = st.file_uploader("Chọn tệp CSV", type="csv")
        col_a, col_b = st.columns(2)
        if uploaded is not None and col_a.button("Làm sạch tệp đã tải lên", type="primary"):
            try:
                st.session_state["df"] = load_and_clean_data(uploaded)
                st.session_state["source"] = uploaded.name
                uploaded.seek(0)
                st.session_state["raw"] = pd.read_csv(uploaded, encoding="utf-8-sig")
                st.rerun()
            except (ValueError, UnicodeDecodeError) as exc:
                st.error(f"Không xử lý được tệp: {exc}")
        if "df" in st.session_state and col_b.button("Quay về dữ liệu mặc định"):
            for key in ("df", "source", "raw"):
                st.session_state.pop(key, None)
            st.rerun()

    raw = st.session_state.get("raw")
    if raw is None:
        raw = pd.read_csv(DEFAULT_DATA, encoding="utf-8-sig")

    # Chỉ số chất lượng dữ liệu
    raw_missing = raw.isna() | raw.astype(str).apply(lambda s: s.str.strip().isin(["", "N/a", "N/A", "nan"]))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Số dòng thô", f"{len(raw):,}")
    c2.metric("Số dòng sau làm sạch", f"{len(df):,}", delta=f"{len(df) - len(raw):,}")
    c3.metric("Ô thiếu (thô)", f"{int(raw_missing.values.sum()):,}")
    c4.metric("Số thể loại", f"{df['Genre'].str.split(',').explode().str.strip().nunique()}")

    tab_raw, tab_clean, tab_quality = st.tabs(["📄 Dữ liệu thô", "✨ Dữ liệu đã làm sạch", "🔍 Chất lượng dữ liệu"])
    with tab_raw:
        st.dataframe(raw.head(200), width="stretch")
        st.caption(f"Hiển thị 200/{len(raw):,} dòng đầu.")
    with tab_clean:
        st.dataframe(df, width="stretch", column_config=money_config(df.columns))
    with tab_quality:
        quality = pd.DataFrame({
            "Kiểu dữ liệu": df.dtypes.astype(str),
            "Giá trị thiếu": df.isna().sum(),
            "Tỷ lệ thiếu (%)": (df.isna().mean() * 100).round(2),
            "Số giá trị khác nhau": df.nunique(),
        })
        st.dataframe(quality, width="stretch")
        st.markdown(
            "**Quy tắc làm sạch đã áp dụng:**\n"
            "- `N/a`, chuỗi rỗng, `None`... → giá trị thiếu thực sự (NaN).\n"
            "- Bỏ ký hiệu `$` và dấu phẩy trong Budget/Revenue, ép về số thực (USD).\n"
            "- Budget/Revenue/Duration ≤ 0 và Rating ngoài [0, 10] → coi là thiếu.\n"
            "- Loại bản ghi trùng theo (Title, Year); **không** điền doanh thu bị thiếu."
        )

    # Lưu kết quả
    df.to_csv(RESULTS_FILE, index=False, encoding="utf-8-sig", na_rep=MISSING_LABEL)
    st.success(f"Đã lưu dữ liệu sạch vào `data/{RESULTS_FILE.name}`.")
    st.download_button(
        "⬇️ Tải results.csv",
        df.to_csv(index=False, na_rep=MISSING_LABEL).encode("utf-8-sig"),
        file_name="results.csv",
        mime="text/csv",
    )


# ----------------------------------------------------------------------------
# PHẦN II
# ----------------------------------------------------------------------------
elif section == "Statistics & Distribution (Part II)":
    st.header("Phần II: Thống kê và phân phối")

    # 1. Thống kê toàn bộ
    st.subheader("1. Thống kê mô tả toàn bộ dữ liệu")
    stats = calculate_basic_stats(df)
    stats.to_csv(STATS_FILE, index=True, encoding="utf-8-sig")
    st.dataframe(
        stats.style.format("{:,.2f}"),
        width="stretch",
    )
    st.caption(f"Đã lưu bảng thống kê vào `data/{STATS_FILE.name}`.")

    # 2. Thống kê theo thể loại
    st.subheader("2. Thống kê theo thể loại")
    genre_col = st.selectbox(
        "Chọn thuộc tính", NUMERIC_COLUMNS, index=3,
        format_func=lambda c: COLUMN_LABELS[c], key="genre_col",
    )
    by_genre = calculate_stats_by_genre(df, genre_col)
    left, right = st.columns([3, 2])
    with left:
        st.dataframe(by_genre.style.format({"Count": "{:,.0f}", "Mean": "{:,.2f}",
                                            "Median": "{:,.2f}", "Std": "{:,.2f}"}),
                     width="stretch")
    with right:
        st.bar_chart(by_genre["Mean"], horizontal=True)
    st.caption("Phim có nhiều thể loại được tính vào tất cả các thể loại của nó; chỉ hiện thể loại có ≥ 5 phim.")

    # 3. Top / Bottom
    st.subheader("3. Top & Bottom phim")
    c1, c2 = st.columns([2, 1])
    tb_col = c1.selectbox(
        "Xếp hạng theo", NUMERIC_COLUMNS, index=1,
        format_func=lambda c: COLUMN_LABELS[c], key="tb_col",
    )
    top_n = c2.slider("Số phim mỗi bảng", 3, 20, 5)
    top, bottom = get_top_bottom_movies(df, tb_col, top_n=top_n)
    left, right = st.columns(2)
    with left:
        st.markdown(f"**🏆 Top {top_n} – cao nhất**")
        st.dataframe(top.reset_index(drop=True), width="stretch",
                     column_config=money_config(top.columns))
    with right:
        st.markdown(f"**📉 Bottom {top_n} – thấp nhất**")
        st.dataframe(bottom.reset_index(drop=True), width="stretch",
                     column_config=money_config(bottom.columns))

    # 4. Histogram
    st.subheader("4. Biểu đồ phân phối (Histogram)")
    c1, c2 = st.columns([2, 1])
    hist_col = c1.selectbox(
        "Chọn thuộc tính", NUMERIC_COLUMNS, index=3,
        format_func=lambda c: COLUMN_LABELS[c], key="hist_col",
    )
    log_scale = c2.checkbox(
        "Thang log10", value=hist_col in MONEY_COLUMNS,
        help="Kinh phí và doanh thu lệch phải mạnh; thang log giúp nhìn rõ phân phối.",
    )
    try:
        show_figure(plot_histogram(df, hist_col, log_scale=log_scale))
        skew = pd.to_numeric(df[hist_col], errors="coerce").skew()
        st.caption(
            f"Độ lệch (skewness) = {skew:.2f} → "
            + ("lệch phải (đuôi dài về phía giá trị lớn)." if skew > 0.5
               else "lệch trái." if skew < -0.5 else "gần đối xứng.")
        )
    except ValueError as exc:
        st.warning(str(exc))


# ----------------------------------------------------------------------------
# PHẦN III
# ----------------------------------------------------------------------------
elif section == "Clustering & Radar Chart (Part III)":
    st.header("Phần III: Phân cụm và biểu đồ radar")

    # 1. Chọn số cụm K
    st.subheader("1. Chọn số cụm K (Elbow và Silhouette)")
    try:
        k_scores = evaluate_k_cached(df)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    elbow_k = k_scores.attrs["elbow_k"]
    best_sil_k = k_scores.attrs["best_k_silhouette"]
    left, mid, right = st.columns([2, 2, 2])
    with left:
        st.markdown("**Elbow: Inertia theo K**")
        st.line_chart(k_scores["Inertia"], x_label="K", y_label="Inertia")
    with mid:
        st.markdown("**Silhouette theo K** (càng cao càng tốt)")
        st.line_chart(k_scores["Silhouette"], x_label="K", y_label="Silhouette")
    with right:
        st.dataframe(k_scores.style.format({"Inertia": "{:,.0f}", "Silhouette": "{:.3f}"}),
                     width="stretch")
    st.caption(
        f"Elbow gợi ý **K = {elbow_k}** (từ đây Inertia giảm chậm lại). "
        f"Silhouette cao nhất ở **K = {best_sil_k}**"
        + (", nhưng K = 2 thường chỉ tách bom tấn khỏi phần còn lại nên ít thông tin. "
           if best_sil_k == 2 else ". ")
        + f"Mặc định dùng K = {elbow_k} theo Elbow."
    )

    # 2. K-Means + PCA
    st.subheader("2. Phân cụm K-Means trên Budget, Revenue, Rating")
    n_clusters = st.slider("Số cụm (K)", 2, 8, elbow_k)
    try:
        clustered = run_kmeans_clustering(df, n_clusters)
        df_pca = run_pca_reduction(df)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    cluster_names = clustered.attrs.get("cluster_names", {})

    left, right = st.columns([3, 2])
    with left:
        show_figure(plot_pca_clusters(df_pca, clustered["Cluster"], names=cluster_names))
    with right:
        st.markdown("**Hồ sơ trung bình của từng cụm**")
        profile = clustered.groupby("Cluster").agg(
            So_phim=("Cluster", "size"),
            Budget_TB=("Budget", "mean"),
            Revenue_TB=("Revenue", "mean"),
            Rating_TB=("Rating", "mean"),
        )
        profile["ROI_TB"] = profile["Revenue_TB"] / profile["Budget_TB"]
        profile.insert(0, "Ten_cum", profile.index.map(cluster_names))
        st.dataframe(
            profile.style.format({
                "So_phim": "{:,.0f}", "Budget_TB": "${:,.0f}",
                "Revenue_TB": "${:,.0f}", "Rating_TB": "{:.2f}", "ROI_TB": "{:.2f}x",
            }),
            width="stretch",
        )
        st.caption(
            f"Chỉ {len(clustered):,}/{len(df):,} phim có đủ Budget, Revenue, Rating được phân cụm. "
            "Cụm được đánh số theo doanh thu trung bình giảm dần (cụm 0 cao nhất). "
            "Tên cụm tự đặt theo doanh thu, kèm \"điểm cao/thấp\" nếu điểm TB lệch "
            "khỏi mức chung quá 0.5 độ lệch chuẩn. ROI_TB = Doanh thu TB / Kinh phí TB."
        )

    # Danh sách phim trong cụm
    chosen = st.selectbox(
        "Xem danh sách phim trong cụm", sorted(clustered["Cluster"].unique()),
        format_func=lambda c: f"Cụm {c}: {cluster_names.get(int(c), '')}",
    )
    members = df.loc[clustered.index[clustered["Cluster"] == chosen],
                     ["Title", "Year", "Genre", "Rating", "Budget", "Revenue"]]
    st.dataframe(members.sort_values("Revenue", ascending=False), width="stretch",
                 column_config=money_config(members.columns), height=280)

    # 3. Radar
    st.divider()
    st.subheader("3. So sánh hai phim bằng biểu đồ Radar")
    radar_df = df.copy()
    year_text = radar_df["Year"].astype("string").fillna("?")
    radar_df["Title"] = radar_df["Title"].astype(str) + " (" + year_text + ")"
    options = radar_df["Title"].tolist()

    def default_index(keyword, fallback):
        matches = [i for i, t in enumerate(options) if t.startswith(keyword)]
        return matches[0] if matches else fallback

    c1, c2 = st.columns(2)
    movie1 = c1.selectbox("Phim thứ nhất", options, index=default_index("Avatar", 0))
    movie2 = c2.selectbox("Phim thứ hai", options, index=default_index("Titanic", min(1, len(options) - 1)))

    if movie1 == movie2:
        st.info("Hãy chọn hai phim khác nhau để so sánh.")
    else:
        left, right = st.columns([3, 2])
        try:
            with left:
                show_figure(plot_radar_chart(radar_df, movie1, movie2))
            with right:
                st.markdown("**Giá trị gốc**")
                compare = radar_df.set_index("Title").loc[
                    [movie1, movie2], ["Rating", "Budget", "Revenue", "Duration", "Genre", "Director"]
                ].T
                st.dataframe(compare.astype(str), width="stretch")
                st.caption("Biểu đồ dùng giá trị đã chuẩn hóa Min-Max về [0, 1] trên toàn bộ dữ liệu.")
        except ValueError as exc:
            st.warning(str(exc))


# ----------------------------------------------------------------------------
# PHẦN IV
# ----------------------------------------------------------------------------
elif section == "Revenue Prediction (Part IV)":
    st.header("Phần IV: Dự đoán doanh thu")

    # Mô hình nhận cột Genre dạng "Action, Drama" và tự tách thành từng thể loại
    try:
        model = train_model_cached(df)
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    metrics = model.metrics_

    # 1. Đánh giá
    st.subheader("1. Kết quả đánh giá mô hình (RandomForestRegressor)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"${metrics['MAE'] / 1e6:,.1f} triệu")
    c2.metric("RMSE", f"${metrics['RMSE'] / 1e6:,.1f} triệu")
    c3.metric("R² Score", f"{metrics['R2']:.3f}")
    c4.metric("Train / Test", f"{metrics['n_train']} / {metrics['n_test']}")
    st.caption(
        f"Đặc trưng đầu vào: {', '.join(model.feature_columns_)}"
        f" (Genre được tách thành {len(model.genre_options_)} cột 0/1, mỗi thể loại một cột). "
        "Mô hình học trên log(1 + Revenue) để giảm ảnh hưởng của vài bom tấn, kết quả đổi lại về USD. "
        "Chia train/test 80/20 (random_state=42); tiền xử lý chỉ học trên tập train. "
        f"R² = {metrics['R2']:.2f} nghĩa là mô hình giải thích khoảng {max(metrics['R2'], 0):.0%} "
        "biến động doanh thu trên tập kiểm tra."
    )

    # Mức độ quan trọng của đặc trưng
    with st.expander("📊 Mức độ quan trọng của các đặc trưng"):
        try:
            st.bar_chart(get_feature_importances(model).head(15), horizontal=True)
        except (AttributeError, ValueError):
            st.info("Không lấy được mức độ quan trọng của đặc trưng.")

    # 2. Dự đoán
    st.divider()
    st.subheader("2. Mô phỏng doanh thu cho một dự án phim mới")
    genres = model.genre_options_
    years = pd.to_numeric(df["Year"], errors="coerce").dropna()
    min_year, max_year = int(years.min()), int(years.max())
    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        budget_m = c1.number_input("Kinh phí dự kiến (triệu USD)", 0.1, 1000.0, 50.0, step=5.0)
        duration = c1.number_input("Thời lượng (phút)", 60, 240, 110, step=5)
        year = c1.number_input("Năm phát hành", min_year, max_year, max_year, step=1,
                               help=f"Dữ liệu huấn luyện chỉ có phim từ {min_year} đến {max_year}.")
        rating = c2.slider("Điểm đánh giá kỳ vọng (/10)", 1.0, 10.0, 6.5, step=0.1)
        chosen_genres = c2.multiselect(
            "Thể loại (chọn một hoặc nhiều)", genres,
            default=[g for g in ("Action", "Adventure") if g in genres],
        )
        submitted = st.form_submit_button("🔮 Dự đoán doanh thu", type="primary")

    if submitted:
        user_values = {
            "Budget": budget_m * 1e6,
            "Rating": rating,
            "Duration": duration,
            "Year": year,
            "Genre": ", ".join(chosen_genres),
            "Votes": float("nan"),
        }
        X_new = pd.DataFrame([{c: user_values[c] for c in model.feature_columns_}])
        predicted = float(model.predict(X_new)[0])
        roi = predicted / (budget_m * 1e6)

        c1, c2, c3 = st.columns(3)
        c1.metric("Doanh thu dự đoán", f"${predicted / 1e6:,.1f} triệu")
        c2.metric("Lãi/Lỗ ước tính", f"${(predicted - budget_m * 1e6) / 1e6:,.1f} triệu")
        c3.metric("Tỷ suất Doanh thu / Kinh phí", f"{roi:.2f}x")
        if roi >= 2.5:
            st.success("Dự án có tiềm năng sinh lời tốt (thường cần ~2.5x kinh phí để hòa vốn sau chi phí phát hành).")
        elif roi >= 1:
            st.warning("Doanh thu vượt kinh phí sản xuất nhưng có thể chưa đủ bù chi phí marketing/phát hành.")
        else:
            st.error("Doanh thu dự đoán thấp hơn kinh phí — rủi ro thua lỗ cao.")
        st.caption(
            f"Sai số trung bình (MAE) của mô hình khoảng ±${metrics['MAE'] / 1e6:,.0f} triệu; "
            "kết quả chỉ mang tính tham khảo."
        )