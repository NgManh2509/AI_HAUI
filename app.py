import os
import cv2
import numpy as np
import streamlit as st

from knn_func import load_knn_from_npz

# =========================
# CẤU HÌNH ĐƯỜNG DẪN
# =========================
try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_ROOT = os.getcwd()

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")
HAAR_DIR = os.path.join(PROJECT_ROOT, "haar")
CASCADE_PATH = os.path.join(HAAR_DIR, "haarcascade_frontalface_default.xml")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_knn.npz")

IMG_SIZE = (100, 100)               # kích thước ảnh mặt để train/predict
TARGET_IMAGES_PER_PERSON = 20       # gợi ý số ảnh nên chụp / người

# =========================
# CẤU HÌNH GIAO DIỆN
# =========================
st.set_page_config(
    page_title="AI HAUI - Hệ thống nhận diện khuôn mặt",
    layout="wide",
    page_icon="📷",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 30px;
            font-weight: 800;
            text-align: center;
            margin-bottom: 0.25rem;
        }
        .sub-title {
            text-align: center;
            font-size: 14px;
            color: #666666;
            margin-bottom: 1.5rem;
        }
        .step-box {
            padding: 1rem 1.2rem;
            border-radius: 0.6rem;
            border: 1px solid #e0e0e0;
            margin-bottom: 1rem;
            background-color: #fafafa;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# LOAD HAAR CASCADE
# =========================
if not os.path.exists(CASCADE_PATH):
    st.error(
        f"Không tìm thấy file Haar Cascade: `{CASCADE_PATH}`.\n"
        "Hãy đặt file `haarcascade_frontalface_default.xml` vào thư mục `haar/`."
    )
    st.stop()

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    st.error("Không load được Haar Cascade.")
    st.stop()

# =========================
# LOAD MODEL KNN TỪ NPZ
# =========================
knn_model = load_knn_from_npz(MODEL_PATH)

# =========================
# HÀM TIỆN ÍCH
# =========================
def _save_image(frame, person_folder_path: str):
    """
    Lưu 1 ảnh (grayscale hoặc BGR) vào dataset/<person>/
    Dùng imencode + write để tránh lỗi Unicode path.
    """
    if not os.path.exists(person_folder_path):
        os.makedirs(person_folder_path, exist_ok=True)

    try:
        is_success, img_encoded = cv2.imencode(".jpg", frame)
        if not is_success:
            return False, "Lỗi mã hóa ảnh (cv2.imencode)"

        # Đếm số file hiện có để đặt tên tiếp theo
        count = len(
            [
                f
                for f in os.listdir(person_folder_path)
                if os.path.isfile(os.path.join(person_folder_path, f))
            ]
        ) + 1

        file_path = os.path.join(person_folder_path, f"{count}.jpg")

        # Ghi file dạng nhị phân
        with open(file_path, "wb") as f:
            f.write(img_encoded.tobytes())

        return True, file_path
    except Exception as e:
        return False, str(e)


def detect_and_crop_face_gray(bgr_image, expand_ratio=0.15):
    """
    - Chuyển sang grayscale
    - Dò mặt bằng Haar
    - Lấy khuôn mặt lớn nhất
    - Mở rộng box một chút cho đỡ sát mặt
    Trả về: face_gray, face_color, (x1, y1, x2, y2)
    """
    gray_frame = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80),
    )

    if len(faces) == 0:
        return None, None, None

    # lấy khuôn mặt có diện tích lớn nhất
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    h_expand = int(h * expand_ratio)
    w_expand = int(w * expand_ratio)

    y1 = max(0, y - h_expand)
    y2 = min(gray_frame.shape[0], y + h + h_expand)
    x1 = max(0, x - w_expand)
    x2 = min(gray_frame.shape[1], x + w + w_expand)

    face_gray = gray_frame[y1:y2, x1:x2].copy()
    face_color = bgr_image[y1:y2, x1:x2].copy()

    return face_gray, face_color, (x1, y1, x2, y2)


def draw_rounded_rectangle(img, top_left, bottom_right, color, thickness=2, radius=15):
    """
    Vẽ khung bo góc bằng line + ellipse.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    radius = int(min(radius, (x2 - x1) / 2, (y2 - y1) / 2))

    cv2.line(img, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(img, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(img, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(img, (x2, y1 + radius), (x2, y2 - radius), color, thickness)

    cv2.ellipse(img, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(img, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(img, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def predict_name_from_gray_face(gray_face):
    """
    Dự đoán tên từ 1 ảnh mặt (grayscale 2D) bằng model KNN tự code.
    """
    if knn_model is None:
        return "Unknown (chưa có model)"

    face_resized = cv2.resize(gray_face, IMG_SIZE)
    feat = face_resized.reshape(-1)  # 10000 chiều

    try:
        pred = knn_model.predict(feat)[0]
        return str(pred)
    except Exception:
        return "Unknown"


# =========================
# PAGE 1: CHỤP ẢNH
# =========================
def page_chup_anh():
    st.markdown('<div class="main-title">HỆ THỐNG NHẬN DIỆN KHUÔN MẶT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Bước 1: Thu thập dữ liệu khuôn mặt (crop + grayscale, lưu vào dataset)</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("📸 Chụp ảnh thu thập dữ liệu")

    person_name = st.text_input(
        "Nhập tên / mã định danh (nên không dấu, không khoảng trắng, ví dụ: Manh, Khang, Nguyen_Manh):",
        "",
    )

    if person_name.strip():
        person_folder_path = os.path.join(DATASET_PATH, person_name.strip())
        current_count = 0
        if os.path.exists(person_folder_path):
            current_count = len(
                [
                    f
                    for f in os.listdir(person_folder_path)
                    if os.path.isfile(os.path.join(person_folder_path, f))
                ]
            )
        st.info(f"Hiện có {current_count} ảnh của `{person_name}` trong dataset.")
        st.progress(min(current_count / TARGET_IMAGES_PER_PERSON, 1.0))
    else:
        st.warning("Hãy nhập tên trước khi chụp ảnh (ưu tiên không dấu để tránh lỗi Unicode).")

    picture = st.camera_input("Chụp ảnh khuôn mặt", key="camera_capture")

    if picture is not None:
        if not person_name.strip():
            st.error("Bạn chưa nhập tên/mã định danh.")
            return

        bytes_data = picture.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        if cv2_img is None or cv2_img.size == 0:
            st.error("Không đọc được ảnh từ camera.")
            return

        face_gray, face_color, box = detect_and_crop_face_gray(cv2_img)
        if face_gray is None:
            st.error("Không tìm thấy khuôn mặt. Hãy chụp lại gần hơn / sáng hơn.")
            return

        person_folder_path = os.path.join(DATASET_PATH, person_name.strip())
        success, path = _save_image(face_gray, person_folder_path)

        if success:
            new_count = len(
                [
                    f
                    for f in os.listdir(person_folder_path)
                    if os.path.isfile(os.path.join(person_folder_path, f))
                ]
            )
            st.success(f"Đã lưu ảnh: {os.path.basename(path)}")
            st.info(f"Tổng số ảnh hiện có của {person_name}: {new_count}")

            st.image(
                cv2.cvtColor(face_color, cv2.COLOR_BGR2RGB),
                caption="Khuôn mặt (màu)",
                use_container_width=True,
            )
            st.image(
                face_gray,
                caption="Khuôn mặt (grayscale) đã lưu",
                use_container_width=True,
            )
        else:
            st.error(f"Lỗi khi lưu ảnh: {path}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Sau khi chụp đủ ảnh cho từng người, chạy `python train_model.py` để tạo data.csv + model_knn.npz.")


# =========================
# PAGE 2: NHẬN DIỆN ẢNH UPLOAD
# =========================
def page_nhan_dien():
    st.markdown('<div class="main-title">NHẬN DIỆN KHUÔN MẶT TỪ ẢNH</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Upload ảnh, hệ thống dò mặt + gán tên bằng KNN tự code</div>',
        unsafe_allow_html=True,
    )

    if knn_model is None:
        st.error(
            "Chưa load được model KNN.\n\n"
            "- Hãy đảm bảo đã chạy `python train_model.py`\n"
            "- File model phải nằm ở: `output/model_knn.npz`"
        )
        return

    uploaded_img = st.file_uploader("Chọn ảnh để nhận diện", type=["jpg", "jpeg", "png"])

    if uploaded_img is None:
        return

    bytes_data = uploaded_img.read()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if cv2_img is None or cv2_img.size == 0:
        st.error("Không đọc được ảnh. Thử lại ảnh khác.")
        return

    gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

    if len(faces) == 0:
        st.warning("Không phát hiện khuôn mặt nào trong ảnh.")
        st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        return

    for (x, y, w, h) in faces:
        face_gray = gray[y:y + h, x:x + w]

        name = predict_name_from_gray_face(face_gray)

        draw_rounded_rectangle(
            cv2_img,
            (x, y),
            (x + w, y + h),
            color=(0, 255, 0),
            thickness=2,
            radius=20,
        )
        cv2.putText(
            cv2_img,
            name,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    st.image(
        cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB),
        caption="Kết quả nhận diện",
        use_container_width=True,
    )


# =========================
# MAIN
# =========================
def main():
    st.sidebar.title("Menu")
    choice = st.sidebar.radio("Chọn chức năng", ["Chụp ảnh", "Nhận diện"])

    if choice == "Chụp ảnh":
        page_chup_anh()
    else:
        page_nhan_dien()


if __name__ == "__main__":
    main()
