"""Phần I: Đọc, chuẩn hóa và làm sạch dữ liệu phim IMDb/TMDB."""


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
    """
    pass