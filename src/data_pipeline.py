"""Phần I: Đọc, chuẩn hóa và làm sạch dữ liệu phim IMDb/TMDB."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Tên cột chuẩn, ánh xạ tên thô và ký hiệu thiếu dữ liệu: xem src/config.py
from src.config import (
    MISSING_TOKENS,
    RENAME_MAP,
    REQUIRED_COLUMNS,
    STANDARD_COLUMNS,
    TEXT_COLUMNS,
)


def _clean_genre(value):
    """Chuyển chuỗi JSON thể loại (TMDB) thành 'Action, Drama'; giữ nguyên nếu đã là văn bản."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip()
    if not text.startswith("["):
        return text or np.nan
    try:
        items = json.loads(text.replace('""', '"'))
        names = [item["name"] for item in items if isinstance(item, dict) and "name" in item]
    except (ValueError, TypeError):
        names = re.findall(r'"name"\s*:\s*"([^"]+)"', text)
    return ", ".join(names) if names else np.nan


def _to_number(series):
    """Bỏ ký hiệu tiền tệ / dấu phẩy rồi ép về số thực; giá trị lỗi -> NaN."""
    cleaned = (
        series.astype("string")
        .str.replace(r"[$,\s]", "", regex=True)
        .replace("", pd.NA)
    )
    return pd.to_numeric(cleaned, errors="coerce").astype("float64")


def load_and_clean_data(raw_file_path):
    """Đọc dữ liệu thô và trả về DataFrame đã được làm sạch.

    Tham số:
        raw_file_path (str hoặc pathlib.Path): Đường dẫn tệp dữ liệu thô.

    Trả về khi hoàn thiện:
        pandas.DataFrame: Dữ liệu với tên cột và kiểu dữ liệu thống nhất.

    Công việc cần triển khai:
        - Xác định định dạng, mã hóa và ánh xạ cột của dữ liệu đầu vào.
        - Dùng pandas để đọc tệp; chuyển "N/a", chuỗi rỗng và các ký hiệu
          thiếu dữ liệu tương đương thành giá trị thiếu thực sự.
        - Làm sạch khoảng trắng trong văn bản, chuẩn hóa tên cột và loại
          bản ghi trùng lặp theo mã phim hoặc khóa phù hợp.
        - Chuyển cột số, tiền tệ và ngày tháng về đúng kiểu dữ liệu;
          thống nhất đơn vị tiền tệ trước khi phân tích doanh thu.
        - Xác định quy tắc xử lý dữ liệu thiếu và giá trị không hợp lệ.
          Không điền doanh thu mục tiêu bị thiếu bằng giá trị suy đoán.
        - Báo lỗi rõ ràng khi tệp không tồn tại hoặc thiếu cột bắt buộc.
        - Trả về dữ liệu; tầng gọi quyết định lưu thành data/results.csv.
          Việc học tham số điền khuyết cho mô hình phải dùng tập huấn luyện.

    Ghi chú cài đặt:
        - Nhận cả đường dẫn lẫn đối tượng tệp (file upload của Streamlit).
        - Quy tắc giá trị không hợp lệ -> NaN: Budget/Revenue <= 0,
          Duration <= 0, Rating ngoài [0, 10], Year ngoài [1880, 2100].
        - Khóa trùng lặp: (Title, Year) không phân biệt hoa thường.
        - Giá trị thiếu giữ là NaN (không điền), sắp xếp Title A-Z, Year giảm dần.
    """
    # 1. Kiểm tra tệp và đọc (thử UTF-8 trước, sau đó latin-1)
    is_file_obj = hasattr(raw_file_path, "read")
    if not is_file_obj and not Path(raw_file_path).is_file():
        raise FileNotFoundError(f"Không tìm thấy tệp dữ liệu: {raw_file_path}")

    df = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            if is_file_obj:
                raw_file_path.seek(0)
            df = pd.read_csv(
                raw_file_path, encoding=encoding,
                na_values=MISSING_TOKENS, keep_default_na=True,
            )
            break
        except UnicodeDecodeError:
            continue
    if df is None:
        raise ValueError("Không đọc được tệp với mã hóa UTF-8 hoặc latin-1.")

    # 2. Chuẩn hóa tên cột
    df.columns = [str(c).strip() for c in df.columns]
    df = df.rename(columns=RENAME_MAP)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {', '.join(missing)}.")
    for optional in set(STANDARD_COLUMNS) - set(REQUIRED_COLUMNS):  # Director, Cast
        if optional not in df.columns:
            df[optional] = np.nan
    df = df[STANDARD_COLUMNS].copy()

    # 3. Làm sạch văn bản: bỏ khoảng trắng thừa, chuỗi rỗng -> NaN
    for col in TEXT_COLUMNS:
        df[col] = (
            df[col].astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .replace("", pd.NA)
        )
    df["Genre"] = df["Genre"].map(_clean_genre, na_action="ignore")

    # 4. Ép kiểu số và loại giá trị không hợp lệ
    for col in ("Budget", "Revenue", "Duration", "Rating", "Year"):
        df[col] = _to_number(df[col])
    df.loc[df["Budget"] <= 0, "Budget"] = np.nan
    df.loc[df["Revenue"] <= 0, "Revenue"] = np.nan  # không suy đoán doanh thu
    df.loc[df["Duration"] <= 0, "Duration"] = np.nan
    df.loc[~df["Rating"].between(0, 10), "Rating"] = np.nan
    df.loc[~df["Year"].between(1880, 2100), "Year"] = np.nan
    df["Year"] = df["Year"].round().astype("Int64")
    df["Duration"] = df["Duration"].round().astype("Int64")

    # 5. Bỏ phim không có tên và bản ghi trùng (Title, Year)
    df = df.dropna(subset=["Title"])
    key = df["Title"].str.lower()
    df = df.loc[~pd.DataFrame({"t": key, "y": df["Year"]}).duplicated()]

    # 6. Sắp xếp và đặt lại chỉ mục
    df = df.sort_values(["Title", "Year"], ascending=[True, False]).reset_index(drop=True)

    # Trả về kiểu object cho cột văn bản để tương thích các thư viện khác
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype("object").where(df[col].notna(), np.nan)
    return df