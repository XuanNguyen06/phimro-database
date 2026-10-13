"""Phần III và IV: Phân cụm, giảm chiều và dự đoán doanh thu phim."""

import numbers

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from src.config import (
    CLUSTER_FEATURES,
    GENRE,
    MODEL_NUMERIC_FEATURES,
    MODEL_TARGET,
    RANDOM_STATE,
    RATING,
    REVENUE,
    find_column,
)

# Tên cột nhãn cụm trong kết quả phân cụm
CLUSTER_COL = "Cluster"


# ============================================================================
# PHẦN III: PHÂN CỤM VÀ GIẢM CHIỀU
# ============================================================================
def _prepare_cluster_features(df):
    """Trích đặc trưng số hợp lệ cho K-means/PCA (không sửa df đầu vào).

    Trả về DataFrame mới gồm Budget, Revenue, Rating (tên chuẩn trong
    src/config.py) đã ép kiểu số, loại hàng thiếu/vô hạn, giữ nguyên chỉ mục
    gốc. Chấp nhận cả tên dài Budget_USD, Revenue_USD, IMDb_Rating.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df phải là pandas.DataFrame.")
    if df.empty:
        raise ValueError("Dữ liệu đầu vào rỗng.")

    columns, missing = [], []
    for canonical in CLUSTER_FEATURES:
        found = find_column(df, canonical)
        if found is None:
            missing.append(canonical)
        else:
            columns.append(found)
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc: {', '.join(missing)}.")

    X = (
        df[columns]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )
    X.columns = CLUSTER_FEATURES  # luôn trả về tên chuẩn
    return X.copy()


def _relabel_by_revenue(X, labels):
    """Đánh lại nhãn cụm theo doanh thu trung bình giảm dần (0 = cao nhất).

    K-means gán nhãn ngẫu nhiên, nên cùng một nhóm phim có thể mang nhãn
    khác nhau giữa các lần chạy. Sắp xếp lại giúp nhãn có ý nghĩa ổn định.
    """
    labels = pd.Series(labels, index=X.index)
    order = X[REVENUE].groupby(labels).mean().sort_values(ascending=False).index
    mapping = {old: new for new, old in enumerate(order)}
    return labels.map(mapping).astype(int)


def name_clusters(clustered):
    """Đặt tên dễ hiểu cho từng cụm dựa trên doanh thu và điểm trung bình.

    Tham số:
        clustered (pandas.DataFrame): Kết quả run_kmeans_clustering.

    Trả về:
        dict[int, str]: Nhãn cụm -> tên, ví dụ {0: "Bom tấn", 1: "Tầm trung"}.

    Quy tắc:
        - Xếp cụm theo doanh thu trung bình: cao nhất là "Bom tấn", thấp nhất
          là "Doanh thu thấp" ("Phổ thông" nếu chỉ có 2 cụm), ở giữa là
          "Tầm trung" (đánh số nếu có nhiều).
        - Thêm "· điểm cao"/"· điểm thấp" nếu điểm trung bình của cụm lệch
          khỏi điểm trung bình chung quá 0.5 độ lệch chuẩn.
    """
    if CLUSTER_COL not in clustered.columns:
        raise ValueError(f"Thiếu cột '{CLUSTER_COL}'.")
    profile = clustered.groupby(CLUSTER_COL)[[REVENUE, RATING]].mean()
    order = profile[REVENUE].sort_values(ascending=False).index.tolist()
    k = len(order)
    rating_mean = clustered[RATING].mean()
    rating_std = clustered[RATING].std(ddof=0)

    names = {}
    for rank, label in enumerate(order):
        if k == 1:
            name = "Tất cả phim"
        elif rank == 0:
            name = "Bom tấn"
        elif k == 2:
            name = "Phổ thông"  # cụm còn lại chứa phần lớn phim
        elif rank == k - 1:
            name = "Doanh thu thấp"
        elif k == 3:
            name = "Tầm trung"
        else:
            name = f"Tầm trung {rank}"

        rating = profile.loc[label, RATING]
        if rating_std > 0 and rating >= rating_mean + 0.5 * rating_std:
            name += " · điểm cao"
        elif rating_std > 0 and rating <= rating_mean - 0.5 * rating_std:
            name += " · điểm thấp"
        names[int(label)] = name
    return names


def run_kmeans_clustering(df, n_clusters):
    """Phân cụm phim bằng K-means trên các đặc trưng số đã chuẩn hóa.

    Tham số:
        df (pandas.DataFrame): Dữ liệu phim đã làm sạch.
        n_clusters (int): Số cụm, từ 1 đến số mẫu hợp lệ.

    Trả về khi hoàn thiện:
        pandas.DataFrame: Bản sao dữ liệu có thêm cột cluster.

    Công việc cần triển khai:
        - Thống nhất danh sách đặc trưng; loại mã phim và nhãn cụm cũ.
        - Kiểm tra số cụm, dữ liệu rỗng và số mẫu phân biệt khả dụng.
        - Xử lý giá trị thiếu, vô hạn và chuẩn hóa bằng StandardScaler.
        - Dùng sklearn.cluster.KMeans; đặt random_state và n_init rõ ràng
          để kết quả có thể tái lập.
        - Gắn nhãn đúng chỉ mục ban đầu; quy định cách biểu diễn các hàng
          không thể phân cụm và không thay đổi dữ liệu đầu vào.

    Ghi chú cài đặt (Tổ 3):
        - Đặc trưng: Budget, Revenue, Rating (src/config.py, chấp nhận tên dài
          Budget_USD, Revenue_USD, IMDb_Rating). Kết quả dùng tên chuẩn.
        - Trả về bảng gồm 3 đặc trưng + cột "Cluster" (nhãn 0, 1, 2...),
          giữ chỉ mục gốc. Các hàng thiếu/vô hạn không thể phân cụm bị loại
          khỏi kết quả (không xuất hiện trong bảng trả về).
        - Nhãn được sắp theo doanh thu trung bình giảm dần: cụm 0 luôn có
          doanh thu trung bình cao nhất. Tên cụm (name_clusters) lưu ở
          kết quả.attrs["cluster_names"].
        - Chuẩn hóa bằng StandardScaler; KMeans(random_state=42, n_init=10).
        - Ném ValueError nếu n_clusters không hợp lệ hoặc lớn hơn số mẫu
          phân biệt khả dụng.
    """
    X = _prepare_cluster_features(df)

    if isinstance(n_clusters, bool) or not isinstance(n_clusters, numbers.Integral):
        raise ValueError("n_clusters phải là số nguyên.")
    n_clusters = int(n_clusters)
    if n_clusters < 1:
        raise ValueError("n_clusters phải >= 1.")
    if len(X) == 0:
        raise ValueError("Không còn dòng hợp lệ sau khi loại giá trị thiếu.")
    if n_clusters > len(X):
        raise ValueError(
            f"n_clusters={n_clusters} lớn hơn số mẫu hợp lệ ({len(X)})."
        )
    n_distinct = len(X.drop_duplicates())
    if n_clusters > n_distinct:
        raise ValueError(
            f"n_clusters={n_clusters} lớn hơn số mẫu phân biệt ({n_distinct})."
        )

    # Đưa tiền và điểm số về cùng thang đo trước khi phân cụm
    X_scaled = StandardScaler().fit_transform(X)

    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
    X[CLUSTER_COL] = _relabel_by_revenue(X, kmeans.fit_predict(X_scaled))
    X.attrs["cluster_names"] = name_clusters(X)
    return X


def evaluate_k_range(df, k_min=2, k_max=8):
    """Tính Inertia (Elbow) và Silhouette cho từng K để chọn số cụm.

    Tham số:
        df (pandas.DataFrame): Dữ liệu phim đã làm sạch.
        k_min, k_max (int): Khoảng K cần thử (mặc định 2..8).

    Trả về:
        pandas.DataFrame: Chỉ mục K, cột Inertia và Silhouette.
        - attrs["best_k_silhouette"]: K có Silhouette cao nhất.
        - attrs["elbow_k"]: K tại "khuỷu tay" của đường Inertia, xác định là
          điểm xa nhất so với đoạn thẳng nối K đầu và K cuối.

    Ý nghĩa:
        - Inertia: tổng bình phương khoảng cách tới tâm cụm, luôn giảm khi K
          tăng. Chọn K mà từ đó Inertia giảm chậm lại (khuỷu tay).
        - Silhouette: từ -1 đến 1, càng cao thì các cụm càng tách biệt.
    """
    X = _prepare_cluster_features(df)
    if k_min < 2:
        raise ValueError("Silhouette cần k_min >= 2.")
    # Silhouette yêu cầu K <= số mẫu phân biệt - 1
    k_max = min(k_max, len(X.drop_duplicates()) - 1)
    if k_max < k_min:
        raise ValueError("Không đủ mẫu phân biệt để đánh giá số cụm.")

    X_scaled = StandardScaler().fit_transform(X)
    rows = []
    for k in range(k_min, k_max + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X_scaled)
        rows.append({
            "K": k,
            "Inertia": float(kmeans.inertia_),
            "Silhouette": float(silhouette_score(X_scaled, kmeans.labels_)),
        })
    result = pd.DataFrame(rows).set_index("K")

    # Khuỷu tay: chuẩn hóa hai trục về [0, 1], tìm điểm xa đường chéo nhất
    ks = result.index.to_numpy(dtype=float)
    inertia = result["Inertia"].to_numpy()
    if len(ks) >= 3 and inertia[0] != inertia[-1]:
        x = (ks - ks[0]) / (ks[-1] - ks[0])
        y = (inertia - inertia[-1]) / (inertia[0] - inertia[-1])
        elbow_k = int(ks[np.argmax(np.abs(1 - x - y))])
    else:
        elbow_k = int(ks[0])

    result.attrs["best_k_silhouette"] = int(result["Silhouette"].idxmax())
    result.attrs["elbow_k"] = elbow_k
    return result


def run_pca_reduction(df):
    """Giảm đặc trưng phim xuống hai chiều để trực quan hóa.

    Tham số:
        df (pandas.DataFrame): Dữ liệu phim, có thể chứa nhãn cluster.

    Trả về khi hoàn thiện:
        pandas.DataFrame: Tọa độ PC1 và PC2, giữ chỉ mục của mẫu hợp lệ.

    Công việc cần triển khai:
        - Chọn đặc trưng số tương ứng với bước phân cụm; loại mã phim
          và cột cluster để nhãn cụm không trở thành đặc trưng.
        - Xử lý giá trị thiếu, vô hạn và chuẩn hóa các đặc trưng.
        - Kiểm tra có ít nhất hai mẫu và hai đặc trưng khả dụng.
        - Dùng sklearn.decomposition.PCA với n_components=2.
        - Giữ liên kết chỉ mục để ghép tọa độ với tên phim và nhãn cụm;
          xem xét tỷ lệ phương sai giải thích khi diễn giải biểu đồ.

    Ghi chú cài đặt (Tổ 3):
        - Nhận df, tự chuẩn hóa bên trong bằng StandardScaler rồi chạy
          PCA(n_components=2, random_state=42) trên Budget, Revenue, Rating
          (cột Cluster, nếu có, không được dùng làm đặc trưng).
        - Hai cột kết quả: "Trục X (PCA 1)" và "Trục Y (PCA 2)", giữ chỉ mục
          gốc của các dòng hợp lệ để ghép với Title/Cluster.
        - Tỷ lệ phương sai giải thích lưu ở
          df_pca.attrs["explained_variance_ratio"] (mảng 2 phần tử).
    """
    X = _prepare_cluster_features(df)
    if len(X) < 2:
        raise ValueError("Cần ít nhất 2 dòng hợp lệ để chạy PCA.")
    if X.shape[1] < 2:
        raise ValueError("Cần ít nhất 2 đặc trưng để chạy PCA.")

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    principal_components = pca.fit_transform(X_scaled)

    df_pca = pd.DataFrame(
        data=principal_components,
        columns=["Trục X (PCA 1)", "Trục Y (PCA 2)"],
        index=X.index,  # GIỮ chỉ mục để ghép tên phim / nhãn cụm
    )
    df_pca.attrs["explained_variance_ratio"] = pca.explained_variance_ratio_
    return df_pca


# ============================================================================
# PHẦN IV: DỰ ĐOÁN DOANH THU
# ============================================================================
def split_genres(value):
    """'Action, Drama' -> ['Action', 'Drama']; giá trị thiếu -> []."""
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return []
    return [g.strip() for g in str(value).split(",") if g.strip()]


class GenreEncoder(TransformerMixin, BaseEstimator):
    """Mã hóa toàn bộ thể loại của phim thành các cột 0/1 (multi-hot).

    Ví dụ "Action, Drama" -> Genre_Action = 1, Genre_Drama = 1, các thể loại
    khác = 0. Khác với OneHotEncoder trên cả chuỗi (mỗi tổ hợp thể loại là
    một nhóm riêng, rất ít phim), cách này giữ được thông tin của từng thể
    loại. Thể loại xuất hiện ít hơn min_frequency phim trong tập huấn luyện
    bị bỏ qua; thể loại lạ khi dự đoán cũng bị bỏ qua.
    """

    def __init__(self, min_frequency=5):
        self.min_frequency = min_frequency

    @staticmethod
    def _to_lists(X):
        column = pd.DataFrame(X).iloc[:, 0]
        return [split_genres(v) for v in column]

    def fit(self, X, y=None):
        counts = pd.Series(
            [g for genres in self._to_lists(X) for g in genres], dtype="object"
        ).value_counts()
        self.genres_ = np.array(
            sorted(counts[counts >= self.min_frequency].index), dtype=object
        )
        return self

    def transform(self, X):
        check_is_fitted(self, "genres_")
        position = {g: i for i, g in enumerate(self.genres_)}
        lists = self._to_lists(X)
        out = np.zeros((len(lists), len(self.genres_)))
        for row, genres in enumerate(lists):
            for g in genres:
                if g in position:
                    out[row, position[g]] = 1.0
        return out

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "genres_")
        return np.array([f"{GENRE}_{g}" for g in self.genres_], dtype=object)


def train_revenue_prediction_model(df):
    """Huấn luyện mô hình hồi quy để dự đoán doanh thu phòng vé.

    Tham số:
        df (pandas.DataFrame): Dữ liệu có doanh thu mục tiêu và đặc trưng.

    Trả về khi hoàn thiện:
        sklearn.pipeline.Pipeline: Bộ tiền xử lý và mô hình đã huấn luyện,
        hỗ trợ predict trên bảng đặc trưng có cùng cấu trúc đầu vào.

    Công việc cần triển khai:
        - Ánh xạ cột doanh thu thực tế sang biến mục tiêu; thống nhất đơn vị.
        - Loại hàng thiếu mục tiêu; chọn đặc trưng có sẵn tại lúc dự đoán,
          không dùng doanh thu hoặc thông tin phát sinh từ doanh thu.
        - Chia tập huấn luyện/kiểm tra trước khi học bước tiền xử lý;
          đặt random_state để tái lập và kiểm tra số mẫu tối thiểu.
        - Dùng Pipeline và ColumnTransformer để điền khuyết, mã hóa cột
          phân loại và chuẩn hóa khi cần; chỉ fit trên tập huấn luyện.
        - Chọn LinearRegression hoặc RandomForestRegressor.
        - Đánh giá MAE, RMSE và R² trên tập kiểm tra; ghi nhận kết quả
          phục vụ báo cáo, rồi trả về Pipeline đã huấn luyện.

    Ghi chú cài đặt (Phần IV):
        - Mục tiêu: Revenue (chấp nhận tên dài Revenue_USD), đơn vị USD.
        - Đặc trưng số: Budget, Rating, Duration, Year, Votes; cột nào không
          có trong df sẽ được bỏ qua. Không dùng Title hay thông tin từ doanh thu.
        - Đặc trưng thể loại: Genre dạng "Action, Drama", được GenreEncoder
          tách thành một cột 0/1 cho mỗi thể loại (dùng toàn bộ thể loại,
          không chỉ thể loại đầu tiên).
        - Chia train/test (80/20, random_state=42) rồi mới fit Pipeline.
        - Số: điền khuyết bằng median + StandardScaler.
        - Mô hình: RandomForestRegressor(n_estimators=200, random_state=42),
          huấn luyện trên log(1 + Revenue) qua TransformedTargetRegressor.
          Doanh thu lệch phải rất mạnh; học trên thang log giúp vài bom tấn
          không chi phối mô hình. predict() vẫn trả về USD.
        - Kết quả đánh giá lưu ở pipeline.metrics_ (dict MAE, RMSE, R2,
          n_train, n_test), danh sách cột đầu vào ở pipeline.feature_columns_
          và danh sách thể loại mô hình nhận biết ở pipeline.genre_options_.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df phải là pandas.DataFrame.")
    if df.empty:
        raise ValueError("Dữ liệu đầu vào rỗng.")

    target_col = find_column(df, MODEL_TARGET)
    if target_col is None:
        raise ValueError(f"Thiếu cột mục tiêu {MODEL_TARGET}.")

    # Đặt tên đầu vào thống nhất (tên chuẩn) cho Pipeline
    data = pd.DataFrame(index=df.index)
    numeric_cols = []
    for name in MODEL_NUMERIC_FEATURES:
        col = find_column(df, name)
        if col is not None:
            data[name] = (
                pd.to_numeric(df[col], errors="coerce")
                .astype("float64")
                .replace([np.inf, -np.inf], np.nan)
            )
            numeric_cols.append(name)
    genre_col = find_column(df, GENRE)
    if genre_col is not None:
        data[GENRE] = df[genre_col]
    if not numeric_cols and genre_col is None:
        raise ValueError("Không có đặc trưng nào khả dụng để huấn luyện.")

    y = (
        pd.to_numeric(df[target_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    valid = y.notna() & (y >= 0)  # không điền doanh thu thiếu bằng giá trị suy đoán
    X, y = data.loc[valid], y.loc[valid]
    if len(X) < 10:
        raise ValueError(
            f"Cần ít nhất 10 dòng có doanh thu hợp lệ (hiện có {len(X)})."
        )

    # Chia train/test TRƯỚC khi học bước tiền xử lý (tránh data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    transformers = []
    if numeric_cols:
        transformers.append((
            "num",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            numeric_cols,
        ))
    if genre_col is not None:
        # Dữ liệu ít thì không lọc thể loại hiếm, tránh mất hết cột
        min_freq = 5 if len(X_train) >= 100 else 1
        transformers.append(("genre", GenreEncoder(min_frequency=min_freq), [GENRE]))
    preprocessor = ColumnTransformer(transformers)

    regressor = TransformedTargetRegressor(
        regressor=RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])
    model.fit(X_train, y_train)

    # Đánh giá trên tập kiểm tra (thang USD gốc)
    y_pred = model.predict(X_test)
    model.metrics_ = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "R2": float(r2_score(y_test, y_pred)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    model.feature_columns_ = numeric_cols + ([GENRE] if genre_col is not None else [])
    if genre_col is not None:
        encoder = model.named_steps["preprocessor"].named_transformers_["genre"]
        model.genre_options_ = list(encoder.genres_)
    else:
        model.genre_options_ = []
    return model


def get_feature_importances(model):
    """Trả về Series mức độ quan trọng của từng đặc trưng (giảm dần)."""
    names = model.named_steps["preprocessor"].get_feature_names_out()
    regressor = model.named_steps["regressor"]
    fitted = getattr(regressor, "regressor_", regressor)  # TransformedTargetRegressor
    return pd.Series(
        fitted.feature_importances_,
        index=[n.split("__", 1)[-1] for n in names],
    ).sort_values(ascending=False)