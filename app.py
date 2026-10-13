"""Giao diện khung cho ứng dụng phân tích và dự đoán dữ liệu phim."""

import streamlit as st

from src.data_pipeline import load_and_clean_data
from src.models import (
    run_kmeans_clustering,
    run_pca_reduction,
    train_revenue_prediction_model,
)
from src.statistics_eda import calculate_basic_stats, get_top_bottom_movies
from src.visualizer import plot_histogram, plot_radar_chart


# Các hàm được nhập sẵn để từng nhóm tích hợp sau khi hoàn thiện.
# Không gọi hàm khung lúc này vì các hàm chỉ có pass và trả về None.
st.set_page_config(
    page_title="Movie Data Analysis & Prediction",
    page_icon="🎬",
    layout="wide",
)

st.title("Phân tích dữ liệu và dự đoán doanh thu phim")
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

if section == "Overview & Data Cleaning (Part I)":
    st.header("Phần I: Tổng quan và làm sạch dữ liệu")
    # Bổ sung tải tệp, xem dữ liệu thô và kiểm tra chất lượng dữ liệu.
    # Lưu dữ liệu sạch trong st.session_state để dùng giữa các trang.
    st.info(
        "Nhóm Phần I: Tích hợp load_and_clean_data(raw_file_path) tại đây. "
        "Hiển thị dữ liệu đã làm sạch và bổ sung lưu data/results.csv."
    )

elif section == "Statistics & Distribution (Part II)":
    st.header("Phần II: Thống kê và phân phối")
    # Kiểm tra dữ liệu sạch đã có trước khi cho phép chọn cột phân tích.
    st.info(
        "Nhóm Phần II: Tích hợp calculate_basic_stats(df), "
        "get_top_bottom_movies(df, column) và plot_histogram(df, column). "
        "Lưu bảng thống kê vào data/results2.csv và dùng st.pyplot cho hình."
    )

elif section == "Clustering & Radar Chart (Part III)":
    st.header("Phần III: Phân cụm và biểu đồ radar")
    # Bổ sung chọn số cụm, chọn hai phim và biểu đồ phân tán PCA theo cụm.
    st.info(
        "Nhóm Phần III: Tích hợp run_kmeans_clustering(df, n_clusters), "
        "run_pca_reduction(df) và "
        "plot_radar_chart(df, movie1_name, movie2_name). "
        "Hiển thị nhãn cụm, tọa độ PCA và biểu đồ so sánh hai phim."
    )

elif section == "Revenue Prediction (Part IV)":
    st.header("Phần IV: Dự đoán doanh thu")
    # Bổ sung nút huấn luyện, biểu mẫu nhập đặc trưng và kết quả đánh giá.
    # Lưu mô hình để không huấn luyện lại sau mỗi lần tương tác giao diện.
    st.info(
        "Nhóm Phần IV: Tích hợp train_revenue_prediction_model(df). "
        "Dùng Pipeline trả về để dự đoán từ đặc trưng người dùng nhập; "
        "hiển thị doanh thu cùng đơn vị và các chỉ số đánh giá mô hình."
    )