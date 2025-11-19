# Ứng dụng thuật toán KNN vào bài toán nhận diện người trong ảnh

## 1. Giới thiệu dự án

Dự án "Ứng dụng thuật toán KNN vào bài toán nhận diện người trong ảnh"
là một hệ thống nhận diện khuôn mặt đơn giản, nhẹ, dễ triển khai, sử
dụng OpenCV, Haar Cascade và thuật toán K-Nearest Neighbors (KNN) tự cài
đặt.

Ứng dụng hỗ trợ: - Thu thập dữ liệu khuôn mặt từ camera\
- Huấn luyện mô hình KNN từ dataset\
- Nhận diện khuôn mặt trong ảnh upload\
- Hiển thị tên người kèm khung bo góc đẹp

## 2. Mục tiêu và lý do thực hiện

### Mục tiêu

-   Xây dựng mô hình nhận diện người đơn giản, dễ hiểu.\
-   Giúp sinh viên nắm được cách hoạt động của thuật toán KNN trong thị
    giác máy tính.\
-   Minh họa pipeline AI: thu thập → xử lý → train → nhận diện.

### Lý do thực hiện

-   KNN là thuật toán cơ bản, dễ triển khai nhưng hiệu quả với dataset
    nhỏ.
-   Tạo công cụ nhận diện nhanh, dùng trong đề tài học tập.

## 3. Tính năng chính

-   Chụp ảnh và tạo dataset.
-   Train mô hình KNN.
-   Nhận diện khuôn mặt từ ảnh upload.


## 4. Cấu trúc project :
 
FaceRec_App/
│
├──haar/
│    └──haarcascade_frontalface_default.xml
│
├── dataset/                 # (Bước 1) Lưu ảnh dataset theo từng người
│   ├── NguyenVanA/          
│   ├── TranVanB/
│   └── ...
│
├── output/                  # (Bước 2) Model đã train, được tạo sau khi chạy train_model.py
│   └── model_knn.nqz           # File mã hóa đặc trưng khuôn mặt
│
├── app.py                   # Giao diện chính bằng Streamlit
│
├── train_model.py           # Script Train / Encode Dataset
│
├── requirements.txt         # Danh sách thư viện cần cài
│
├── knn_func.py              # Thuật toán KNN
│
├──data.csv                  # Được tạo sau khi chạy train_model.py
│
└── README.md                # Tài liệu hướng dẫn


## 5. Cách triển khai : 
### Step 1:
        Mở một terminal mới : python -m venv venv 
        Để tạo môi trường ảo cho python 

### Step 2: 
        Mở một terminal khác: pip install streamlit opencv-python numpy Pillow pandas
        Để tải các thư viện cần thiết

### Step 3:
        Chạy chương trình qua lệnh : streamlit run app.py 
        Để chạy webApp qua lib streamlit, chương trình sẽ chuyển qua web streamlit để chạy
        Vào mục đầu tiên, nhập tên và chụp ảnh ( khoảng 20++ bức là đủ) 

### Step 4:
        Tạo terminal mới
        Chạy training model qua lệnh : python train.py
        Chương trình sẽ tạo data.csv và trained_model ở phần output (model_knn.nqz)

### Step 5:
        Quay lại web streamlit vừa mở, upload ảnh để thuật toán chạy và hiện tên đối tượng trong ảnh