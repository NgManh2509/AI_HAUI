import streamlit as st
import cv2
import os
import numpy as np

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

if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH, exist_ok=True)

TARGET_IMAGES_PER_PERSON = 20  # số ảnh gợi ý nên chụp / người

# =========================
# CẤU HÌNH GIAO DIỆN CHUNG
# =========================
st.set_page_config(
    page_title="AI HAUI - Hệ thống Nhận diện khuôn mặt",
    layout="wide",
    page_icon="📷",
)

# CSS nhẹ cho đẹp
st.markdown(
    """
    <style>
        .main-title {
            font-size: 32px;
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
        .step-box, .dataset-box {
            padding: 1rem 1.2rem;
            border-radius: 0.6rem;
            border: 1px solid #e0e0e0;
            margin-bottom: 1rem;
        }
        .step-box {
            background-color: #fafafa;
        }
        .dataset-box {
            background-color: #ffffff;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# KIỂM TRA & LOAD HAAR CASCADE
# =========================
if not os.path.exists(CASCADE_PATH):
    st.error(
        f"Không tìm thấy file cascade: `{CASCADE_PATH}`.\n\n"
        "Hãy tải file **haarcascade_frontalface_default.xml** từ OpenCV và đặt vào thư mục `haar/`."
    )
    st.stop()

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
if face_cascade.empty():
    st.error("Không load được Haar Cascade. Kiểm tra lại file `haarcascade_frontalface_default.xml`.")
    st.stop()


# =========================
# HÀM LƯU ẢNH
# =========================
def _save_image(frame, person_folder_path: str):
    """
    Lưu ảnh (grayscale hoặc màu) vào thư mục dataset/<person>/.
    frame: numpy array (2D grayscale hoặc 3D BGR)
    """
    if not os.path.exists(person_folder_path):
        os.makedirs(person_folder_path, exist_ok=True)

    try:
        is_success, img_encoded = cv2.imencode(".jpg", frame)
        if is_success:
            count = len(
                [
                    f
                    for f in os.listdir(person_folder_path)
                    if os.path.isfile(os.path.join(person_folder_path, f))
                ]
            ) + 1
            file_path = os.path.join(person_folder_path, f"{count}.jpg")

            with open(file_path, "wb") as f:
                f.write(img_encoded.tobytes())

            print(f"Đã lưu ảnh: {file_path}")
            return True, file_path
        else:
            print("Lỗi: cv2.imencode() thất bại.")
            return False, "Lỗi mã hóa ảnh"
    except Exception as e:
        print(f"Lỗi hệ thống khi lưu file: {e}")
        return False, str(e)


# =========================
# HÀM PHÁT HIỆN & CROP KHUÔN MẶT
# =========================
def detect_and_crop_face_gray(bgr_image, expand_ratio=0.15):
    """
    - Chuyển ảnh sang grayscale
    - Dò mặt bằng Haar trên ảnh grayscale
    - Crop vùng mặt (grayscale) + trả thêm bản màu để preview
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

    # Lấy khuôn mặt lớn nhất (tránh trường hợp có nhiều người trong ảnh)
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])

    # Mở rộng box một chút cho đỡ cắt sát mặt
    h_expand = int(h * expand_ratio)
    w_expand = int(w * expand_ratio)

    y1 = max(0, y - h_expand)
    y2 = min(gray_frame.shape[0], y + h + h_expand)
    x1 = max(0, x - w_expand)
    x2 = min(gray_frame.shape[1], x + w + w_expand)

    face_gray = gray_frame[y1:y2, x1:x2].copy()
    face_color = bgr_image[y1:y2, x1:x2].copy()

    return face_gray, face_color, (x1, y1, x2, y2)


# =========================
# PAGE 1: CHỤP ẢNH (THU THẬP DỮ LIỆU)
# =========================
def page_chup_anh():
    st.markdown('<div class="main-title">HỆ THỐNG NHẬN DIỆN KHUÔN MẶT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Bước 1: Thu thập dữ liệu khuôn mặt (crop & grayscale, lưu vào dataset)</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("📸 Bước 1: Thu thập dữ liệu khuôn mặt")

    st.write(
        "- Nhập **tên người** (hoặc mã SV, mã nhân viên, …)\n"
        "- Chụp nhiều ảnh với các góc: **thẳng**, **nghiêng trái**, **nghiêng phải**, **biểu cảm khác nhau**.\n"
        f"- Khuyến nghị: khoảng **10–{TARGET_IMAGES_PER_PERSON} ảnh/người** để train model tốt hơn."
    )

    person_name = st.text_input("Nhập tên / mã định danh của bạn:", "TenNguoiMau")

    # Thông tin số ảnh hiện có của người này
    person_folder_path = (
        os.path.join(DATASET_PATH, person_name.strip())
        if person_name.strip()
        else None
    )
    current_count = 0
    if person_folder_path and os.path.exists(person_folder_path):
        current_count = len(
            [
                f
                for f in os.listdir(person_folder_path)
                if os.path.isfile(os.path.join(person_folder_path, f))
            ]
        )

    if person_name and person_name.strip() and person_name != "TenNguoiMau":
        st.info(f"Hiện tại đã có **{current_count} ảnh** của `{person_name}` trong dataset.")
        progress = min(current_count / TARGET_IMAGES_PER_PERSON, 1.0)
        st.progress(progress)
        st.caption(f"Mục tiêu đề xuất: {TARGET_IMAGES_PER_PERSON} ảnh / người")
    else:
        st.warning("Vui lòng nhập tên/mã định danh thực tế trước khi chụp ảnh.")

    picture = st.camera_input(
        "Chụp ảnh (Thẳng, Nghiêng trái, Nghiêng phải)",
        key="camera_capture",
    )

    if picture is not None:
        if not person_name or person_name == "TenNguoiMau" or person_name.strip() == "":
            st.error("❌ Bạn chưa nhập tên/mã định danh. Vui lòng nhập trước khi chụp!")
        else:
            # Decode ảnh từ camera
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            if cv2_img is not None and cv2_img.size > 0:
                # Phát hiện & crop khuôn mặt (grayscale + preview màu)
                face_gray, face_color, box = detect_and_crop_face_gray(cv2_img)

                if face_gray is None:
                    st.error("Không tìm thấy khuôn mặt trong ảnh. Hãy chụp lại, căn mặt rõ hơn.")
                else:
                    # Lưu ảnh grayscale
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

                        st.success(f"✅ Đã lưu ảnh khuôn mặt (grayscale): **{os.path.basename(path)}**")
                        st.info(f"Tổng số ảnh hiện có của **{person_name}**: **{new_count}**")

                        # Hiển thị preview
                        st.write("📷 Khuôn mặt (màu) để xem rõ:")
                        st.image(cv2.cvtColor(face_color, cv2.COLOR_BGR2RGB), use_container_width=True)

                        st.write("🖤 Khuôn mặt (grayscale) đã lưu:")
                        st.image(face_gray, use_container_width=True)

                        st.caption("👉 Tiếp tục chụp thêm ảnh với nhiều góc khác nhau để dataset đa dạng hơn.")
                    else:
                        st.error(f"❌ Lỗi khi lưu ảnh: {path}")
            else:
                st.error("Không thể đọc dữ liệu ảnh từ camera.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption(
        "AI HAUI – Giai đoạn 1: Thu thập dataset khuôn mặt (crop + grayscale) để train model KNN / face_recognition."
    )


# =========================
# PAGE 2: NHẬN DIỆN
# =========================
def page_nhan_dien():
    st.markdown('<div class="main-title">HỆ THỐNG NHẬN DIỆN KHUÔN MẶT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Bước 2: Nhận diện khuôn mặt từ camera / ảnh upload</div>',
        unsafe_allow_html=True,
    )

    st.subheader("👀 Nhận diện (demo)")
    st.info(
        "Phần này bạn có thể:\n"
        "- Load model đã train (KNN, LBPH, hoặc face_recognition)\n"
        "- Mở camera hoặc upload ảnh, dò mặt và gán tên theo dataset.\n\n"
        "Hiện tại mình chỉ tạo sẵn khung giao diện, bạn nhét code nhận diện của bạn vào đây."
    )

    # Ví dụ khung upload ảnh để nhận diện
    uploaded_img = st.file_uploader("Upload ảnh để nhận diện khuôn mặt", type=["jpg", "jpeg", "png"])

    if uploaded_img is not None:
        bytes_data = uploaded_img.read()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

        if cv2_img is None or cv2_img.size == 0:
            st.error("Không đọc được ảnh. Thử lại ảnh khác.")
            return

        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))

        if len(faces) == 0:
            st.warning("Không phát hiện khuôn mặt nào trong ảnh.")
        else:
            # Vẽ bounding box demo (chưa gắn tên)
            for (x, y, w, h) in faces:
                cv2.rectangle(cv2_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

            st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), caption="Ảnh với bounding box khuôn mặt", use_container_width=True)
            st.caption("👉 Sau này bạn dùng model nhận diện để gán tên vào từng khuôn mặt.")


# =========================
# MAIN: MENU BAR
# =========================
def main():
    # Sidebar menu
    st.sidebar.title("Menu")
    choice = st.sidebar.radio(
        "Chọn chức năng",
        ["Chụp ảnh", "Nhận diện"]
    )

    if choice == "Chụp ảnh":
        page_chup_anh()
    else:
        page_nhan_dien()


if __name__ == "__main__":
    main()
