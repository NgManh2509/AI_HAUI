🎯 Mục tiêu

Nhận diện nhiều khuôn mặt trong một ảnh bất kỳ.

Cho phép người dùng tự tạo dataset bằng ảnh chụp cá nhân.

Giao diện chạy bằng Streamlit, dễ dàng upload ảnh và xem kết quả.

🧪 Tiêu chuẩn hoàn thành

✔ Nhận diện đúng nhiều khuôn mặt trong ảnh
✔ Hỗ trợ tạo dataset bằng ảnh chụp (10–20 ảnh/người)
✔ Có mô-đun train để tạo file mã hoá khuôn mặt (encodings.pickle)

🔄 Quy trình thực hiện
Bước 1: Tạo Dataset

Chụp 10–20 ảnh cho mỗi người.

Lưu ảnh theo từng folder con tương ứng với tên mỗi người.

Bước 2: Upload ảnh kiểm thử

Mở giao diện Streamlit.

Upload ảnh và để hệ thống tự động nhận diện.

Bước 3: Testing

Kiểm tra kết quả nhận diện.

Có thể chụp thêm ảnh và cải thiện dataset nếu cần.

🛠 Công nghệ sử dụng

Ngôn ngữ: Python

Thư viện chính:

face_recognition – Nhận diện khuôn mặt

streamlit – Giao diện web chạy trực tiếp

📌 Yêu cầu môi trường

Python 3.10+

Cài đặt thư viện bằng:

pip install -r requirements.txt


Cấu trúc project :
 
FaceRec_App/
│
├── dataset/                 # (Bước 1) Lưu ảnh dataset theo từng người
│   ├── NguyenVanA/          
│   ├── TranVanB/
│   └── ...
│
├── output/                  # (Bước 2) Model đã train
│   └── encodings.pickle     # File mã hóa đặc trưng khuôn mặt
│
├── app.py                   # Giao diện chính bằng Streamlit
│
├── train_model.py           # Script Train / Encode Dataset
│
├── requirements.txt         # Danh sách thư viện cần cài
│
└── README.md                # Tài liệu hướng dẫn
