"""Script dòng lệnh: làm sạch tệp dữ liệu thô và ghi ra CSV.

Toàn bộ quy tắc làm sạch nằm trong src/data_pipeline.load_and_clean_data
(cũng là hàm mà giao diện Streamlit dùng), nên script này chỉ đọc, gọi hàm
và ghi kết quả. Nhờ vậy chỉ cần sửa quy tắc ở một nơi.

Cách chạy (từ thư mục gốc dự án):
    python -m src.data_clean                                  # mặc định
    python -m src.data_clean resultsRaw_completed.csv -o data/results.csv
"""

import argparse
import sys
from pathlib import Path

# Cho phép chạy trực tiếp "python src/data_clean.py" (không qua -m)
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DEFAULT_DATA, MISSING_LABEL, RESULTS_FILE  # noqa: E402
from src.data_pipeline import load_and_clean_data  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description="Làm sạch dữ liệu phim IMDb/TMDB.")
    parser.add_argument(
        "input", nargs="?", default=str(DEFAULT_DATA),
        help=f"Tệp CSV thô (mặc định: {DEFAULT_DATA.name})",
    )
    parser.add_argument(
        "-o", "--output", default=str(RESULTS_FILE),
        help=f"Tệp CSV kết quả (mặc định: data/{RESULTS_FILE.name})",
    )
    args = parser.parse_args(argv)

    print(f"Đang đọc và làm sạch {args.input}...")
    try:
        df = load_and_clean_data(args.input)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 1

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Ô trống ghi thành "N/a" theo quy ước của đề bài
    df.to_csv(output, index=False, encoding="utf-8-sig", na_rep=MISSING_LABEL)
    print(f"Đã tạo {output} ({len(df):,} phim, {df.isna().sum().sum():,} ô thiếu).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
