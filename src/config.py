"""Cấu hình dùng chung: tên cột chuẩn, tên gọi khác và đường dẫn dữ liệu.

Mọi module (làm sạch, thống kê, mô hình, biểu đồ, giao diện) lấy tên cột từ
đây để không phải tự dò "Budget" hay "Budget_USD" ở nhiều nơi.
"""

from pathlib import Path

# ----------------------------------------------------------------------------
# Đường dẫn
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DEFAULT_DATA = DATA_DIR / "phim.csv"     # dữ liệu đầu vào mặc định
RESULTS_FILE = DATA_DIR / "results.csv"  # dữ liệu sau làm sạch (Phần I)
STATS_FILE = DATA_DIR / "results2.csv"   # bảng thống kê (Phần II)

# ----------------------------------------------------------------------------
# Tên cột chuẩn (9 thuộc tính theo README mục 6)
# ----------------------------------------------------------------------------
TITLE = "Title"
YEAR = "Year"
GENRE = "Genre"
DURATION = "Duration"
RATING = "Rating"
BUDGET = "Budget"
REVENUE = "Revenue"
DIRECTOR = "Director"
CAST = "Cast"
VOTES = "Votes"  # README có nhắc, dữ liệu hiện tại chưa có

STANDARD_COLUMNS = [TITLE, YEAR, GENRE, DURATION, RATING, BUDGET, REVENUE, DIRECTOR, CAST]
# Cột bắt buộc sau khi đổi tên; Director/Cast được phép thiếu
REQUIRED_COLUMNS = [TITLE, YEAR, GENRE, DURATION, RATING, BUDGET, REVENUE]
TEXT_COLUMNS = [TITLE, GENRE, DIRECTOR, CAST]
NUMERIC_COLUMNS = [DURATION, RATING, BUDGET, REVENUE]
MONEY_COLUMNS = {BUDGET, REVENUE}

# Tên chuẩn -> các tên chấp nhận được (tên chuẩn đứng đầu, sau đó là tên
# trong dữ liệu thô resultsRaw_completed.csv / README mục 3.1)
COLUMN_ALIASES = {
    TITLE: (TITLE,),
    YEAR: (YEAR, "Release_Year"),
    GENRE: (GENRE,),
    DURATION: (DURATION, "Duration_min"),
    RATING: (RATING, "IMDb_Rating"),
    BUDGET: (BUDGET, "Budget_USD"),
    REVENUE: (REVENUE, "Revenue_USD"),
    DIRECTOR: (DIRECTOR,),
    CAST: (CAST, "Actors"),
    VOTES: (VOTES,),
}

# Tên thô -> tên chuẩn, dùng khi đọc dữ liệu (Phần I)
RENAME_MAP = {
    alias: canonical
    for canonical, aliases in COLUMN_ALIASES.items()
    for alias in aliases
    if alias != canonical
}

# Nhãn hiển thị trên giao diện
COLUMN_LABELS = {
    DURATION: "Thời lượng (phút)",
    RATING: "Điểm đánh giá (/10)",
    BUDGET: "Kinh phí (USD)",
    REVENUE: "Doanh thu (USD)",
}

# Các ký hiệu được coi là thiếu dữ liệu
MISSING_TOKENS = ["", "N/a", "N/A", "n/a", "NA", "NaN", "nan", "None", "null", "NULL", "-"]
MISSING_LABEL = "N/a"  # ký hiệu ghi ra file CSV kết quả

# ----------------------------------------------------------------------------
# Đặc trưng cho phân cụm và mô hình
# ----------------------------------------------------------------------------
CLUSTER_FEATURES = [BUDGET, REVENUE, RATING]
MODEL_NUMERIC_FEATURES = [BUDGET, RATING, DURATION, YEAR, VOTES]
MODEL_TARGET = REVENUE
RADAR_ATTRIBUTES = [RATING, VOTES, BUDGET, REVENUE, DURATION]
RANDOM_STATE = 42


def find_column(df, canonical):
    """Trả về tên cột thực tế trong df ứng với tên chuẩn, hoặc None nếu không có."""
    for name in COLUMN_ALIASES.get(canonical, (canonical,)):
        if name in df.columns:
            return name
    return None
