"""Phần II và III: Tạo biểu đồ Matplotlib để hiển thị trên Streamlit."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    BUDGET,
    COLUMN_ALIASES,
    DURATION,
    MONEY_COLUMNS,
    RADAR_ATTRIBUTES,
    RATING,
    REVENUE,
    TITLE,
    find_column,
)

# Đơn vị hiển thị trên trục của histogram (tên chuẩn và tên dài đều dùng được)
_UNIT_BY_CANONICAL = {
    BUDGET: "triệu USD",
    REVENUE: "triệu USD",
    DURATION: "phút",
    RATING: "điểm /10",
}
_UNITS = {
    alias: unit
    for canonical, unit in _UNIT_BY_CANONICAL.items()
    for alias in COLUMN_ALIASES[canonical]
}
_MONEY_COLUMNS = {alias for c in MONEY_COLUMNS for alias in COLUMN_ALIASES[c]}


def plot_histogram(df, column, log_scale=False):
    """Tạo biểu đồ phân phối của một cột số.

    Tham số:
        df (pandas.DataFrame): Dữ liệu phim đã làm sạch.
        column (str): Tên cột số cần trực quan hóa.
        log_scale (bool): Vẽ trục X theo thang log10 (hữu ích cho tiền tệ
            bị lệch phải mạnh). Mặc định False.

    Trả về khi hoàn thiện:
        matplotlib.figure.Figure: Hình để tầng giao diện dùng st.pyplot.

    Công việc cần triển khai:
        - Kiểm tra cột tồn tại, có kiểu số và còn giá trị hợp lệ.
        - Bỏ giá trị thiếu hoặc vô hạn, chọn số khoảng chia phù hợp.
        - Tạo Figure/Axes riêng bằng matplotlib.pyplot.subplots.
        - Vẽ histogram với tiêu đề, tên trục và đơn vị rõ ràng.
        - Trả về Figure, không gọi plt.show hoặc hàm Streamlit tại đây;
          tầng gọi chịu trách nhiệm đóng hình sau khi sử dụng.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df phải là pandas.DataFrame.")
    if column not in df.columns:
        raise ValueError(f"Không có cột '{column}' trong dữ liệu.")

    values = (
        pd.to_numeric(df[column], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    if log_scale:
        values = values[values > 0]
    if values.empty:
        raise ValueError(f"Cột '{column}' không có giá trị số hợp lệ để vẽ.")

    is_money = column in _MONEY_COLUMNS
    unit = _UNITS.get(column, "")
    # Tiền tệ hiển thị theo triệu USD cho dễ đọc
    plot_values = values / 1e6 if is_money else values

    if log_scale:
        data = np.log10(plot_values)
        x_label = f"log10({column}{' - ' + unit if unit else ''})"
    else:
        data = plot_values
        x_label = f"{column} ({unit})" if unit else column

    # Số khoảng chia tự động (Freedman–Diaconis/Sturges), giới hạn 10..50
    edges = np.histogram_bin_edges(data, bins="auto")
    bins = int(np.clip(len(edges) - 1, 10, 50))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(data, bins=bins, color="#4C72B0", edgecolor="white", alpha=0.85)

    mean, median = data.mean(), data.median()
    ax.axvline(mean, color="#DD8452", linestyle="--", linewidth=2, label=f"Mean = {mean:,.2f}")
    ax.axvline(median, color="#55A868", linestyle="-", linewidth=2, label=f"Median = {median:,.2f}")

    ax.set_title(f"Phân phối {column} (n = {len(values):,})")
    ax.set_xlabel(x_label)
    ax.set_ylabel("Số lượng phim")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pca_clusters(df_pca, labels, titles=None, names=None):
    """Vẽ biểu đồ phân tán 2D các phim theo tọa độ PCA, tô màu theo cụm.

    Tham số:
        df_pca (pandas.DataFrame): Kết quả run_pca_reduction (2 cột tọa độ).
        labels (pandas.Series): Nhãn cụm, cùng chỉ mục với df_pca.
        titles (pandas.Series, tùy chọn): Không dùng để vẽ, giữ cho mở rộng.
        names (dict, tùy chọn): Nhãn cụm -> tên hiển thị trong chú giải
            (kết quả models.name_clusters).

    Trả về:
        matplotlib.figure.Figure
    """
    x_col, y_col = df_pca.columns[:2]
    labels = labels.reindex(df_pca.index)
    ratio = df_pca.attrs.get("explained_variance_ratio")
    names = names or {}

    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for i, cluster in enumerate(sorted(labels.dropna().unique())):
        mask = labels == cluster
        name = names.get(int(cluster))
        label = f"Cụm {int(cluster)}" + (f": {name}" if name else "") + f" ({mask.sum()})"
        ax.scatter(
            df_pca.loc[mask, x_col], df_pca.loc[mask, y_col],
            s=25, alpha=0.7, color=cmap(i % 10), label=label,
        )

    if ratio is not None and len(ratio) >= 2:
        ax.set_xlabel(f"{x_col} – giải thích {ratio[0]:.1%} phương sai")
        ax.set_ylabel(f"{y_col} – giải thích {ratio[1]:.1%} phương sai")
    else:
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    ax.set_title("Bản đồ phân cụm phim (K-Means + PCA 2D)")
    ax.grid(alpha=0.3)
    ax.legend(title="Nhãn cụm")
    fig.tight_layout()
    return fig



def plot_radar_chart(df, movie1_name, movie2_name):
    """So sánh đặc trưng của hai phim trên biểu đồ radar.

    Tham số:
        df (pandas.DataFrame): Dữ liệu có tên phim và các đặc trưng số.
        movie1_name (str): Tên phim thứ nhất.
        movie2_name (str): Tên phim thứ hai.

    Trả về khi hoàn thiện:
        matplotlib.figure.Figure: Biểu đồ radar so sánh hai phim.

    Công việc cần triển khai:
        - Thống nhất cột tên phim; báo lỗi nếu không tìm thấy hoặc trùng
          tên nhiều bản ghi, yêu cầu phân biệt bằng năm phát hành/mã phim.
        - Chọn ít nhất ba đặc trưng số có ý nghĩa; xử lý giá trị thiếu.
        - Chuẩn hóa về cùng thang đo dựa trên tập dữ liệu tham chiếu,
          không chỉ hai phim; xử lý riêng đặc trưng có giá trị hằng.
        - Tạo trục cực, khép kín các đa giác, thêm nhãn và chú giải;
          ghi rõ các giá trị hiển thị đã được chuẩn hóa.
        - Trả về Figure, không gọi plt.show hoặc sửa dữ liệu đầu vào;
          tầng gọi đóng hình sau khi hiển thị.

    Ghi chú cài đặt (Tổ 3):
        - Cột tên phim là Title; so khớp chính xác (đã bỏ khoảng trắng thừa).
          Ném ValueError nếu không tìm thấy hoặc trùng nhiều bản ghi.
        - Thuộc tính: Rating, Votes, Budget, Revenue, Duration (RADAR_ATTRIBUTES
          trong src/config.py, chấp nhận tên dài IMDb_Rating, Budget_USD,
          Revenue_USD, Duration_min). Thuộc tính thiếu cột hoặc thiếu giá trị
          ở một trong hai phim bị bỏ qua; cần còn ít nhất 3 thuộc tính.
        - MinMaxScaler được học trên toàn bộ df (tập tham chiếu), không chỉ
          hai phim; thuộc tính hằng số được đặt ở mức 0.5.
        - Các giá trị trên biểu đồ đã chuẩn hóa về [0, 1], không phải giá trị
          gốc.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df phải là pandas.DataFrame.")
    if TITLE not in df.columns:
        raise ValueError(f"Thiếu cột '{TITLE}' để tìm tên phim.")

    # 1. Lọc dòng dữ liệu của từng phim (đúng thứ tự phim 1, phim 2)
    titles = df[TITLE].astype(str).str.strip()
    rows = []
    for name in (movie1_name, movie2_name):
        matched = df[titles == str(name).strip()]
        if len(matched) == 0:
            raise ValueError(f"Không tìm thấy phim: '{name}'.")
        if len(matched) > 1:
            raise ValueError(
                f"Có {len(matched)} bản ghi trùng tên '{name}'; "
                "hãy phân biệt bằng năm phát hành hoặc mã phim."
            )
        rows.append(matched.index[0])

    # 2. Chọn thuộc tính so sánh (cột số hợp lệ, tập tham chiếu = toàn bộ df)
    reference = {}
    for canonical in RADAR_ATTRIBUTES:
        found = find_column(df, canonical)
        if found is not None:
            reference[canonical] = (
                pd.to_numeric(df[found], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
            )
    reference = pd.DataFrame(reference)
    pair = reference.loc[rows]
    attributes = [a for a in reference.columns if pair[a].notna().all()]
    if len(attributes) < 3:
        raise ValueError(
            "Cần ít nhất 3 thuộc tính có đủ dữ liệu cho cả hai phim "
            f"(hiện có: {', '.join(attributes) or 'không có'})."
        )

    # 3. Min-Max Scaling về [0, 1] dựa trên toàn bộ dữ liệu tham chiếu
    scaler = MinMaxScaler()
    scaler.fit(reference[attributes].dropna(how="all"))
    movies_scaled = scaler.transform(pair[attributes])
    constant = scaler.data_range_ == 0
    movies_scaled[:, constant] = 0.5

    # 4. Góc các đỉnh radar + đóng vòng tròn
    angles = np.linspace(0, 2 * np.pi, len(attributes), endpoint=False).tolist()
    angles += angles[:1]

    # 5. Vẽ
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    for row, label in zip(movies_scaled, (movie1_name, movie2_name)):
        values = row.tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, label=label)
        ax.fill(angles, values, alpha=0.25)

    ax.set_ylim(0, 1)
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(attributes)
    ax.set_title("So sánh hai phim (giá trị đã chuẩn hóa Min-Max về [0, 1])", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    return fig