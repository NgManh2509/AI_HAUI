import os
import cv2
import numpy as np
import streamlit as st

from PIL import ImageFont, ImageDraw, Image
from knn_func import load_knn_from_npz
from deepface import DeepFace

# =========================
# CẤU HÌNH ĐƯỜNG DẪN
# =========================
try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_ROOT = os.getcwd()

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_knn.npz")

# DeepFace config
MODEL_NAME = "ArcFace"  # Chính xác hơn Facenet
DETECTOR_BACKEND = "retinaface"
TARGET_IMAGES_PER_PERSON = 10      

# =========================
# CẤU HÌNH GIAO DIỆN
# =========================
st.set_page_config(
    page_title="AI HAUI - Face Recognition with DeepFace",
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
# LOAD MODEL KNN
# =========================
knn_model = load_knn_from_npz(MODEL_PATH)

# =========================
# HÀM TIỆN ÍCH
# =========================
def _save_image(frame, person_folder_path: str):
    """Lưu ảnh BGR vào dataset (hỗ trợ Unicode path)"""
    if not os.path.exists(person_folder_path):
        os.makedirs(person_folder_path, exist_ok=True)

    try:
        is_success, img_encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not is_success:
            return False, "Lỗi mã hóa ảnh"

        count = len([f for f in os.listdir(person_folder_path) 
                    if os.path.isfile(os.path.join(person_folder_path, f))]) + 1

        file_path = os.path.join(person_folder_path, f"{count}.jpg")

        with open(file_path, "wb") as f:
            f.write(img_encoded.tobytes())

        return True, file_path
    except Exception as e:
        return False, str(e)


def detect_and_extract_faces(bgr_image):
    try:
        rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        #Nhận diện khuôn mặt
        face_objs = DeepFace.extract_faces(
            img_path=rgb_image,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True
        )
        
        if not face_objs:
            return []
        
        results = []
        # Trả về danh sách (face_bgr, facial_area)
        for face_obj in face_objs:
            facial_area = face_obj['facial_area']
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            
            face_bgr = bgr_image[y:y+h, x:x+w]
            results.append((face_bgr, facial_area))
        
        return results
    except Exception as e:
        print(f"Lỗi detect face: {e}")
        return []


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


def predict_name_from_face(face_bgr, confidence_threshold=3.7):
    if knn_model is None:
        return "Unknown", None

    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        
        result = DeepFace.represent(
            img_path=face_rgb,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False
        )
        
        embedding = result[0]["embedding"]
        pred = knn_model.predict(embedding)[0]
        
        # Tính distance để đánh giá confidence
        X_train = knn_model.X_train
        distances = np.linalg.norm(X_train - np.array(embedding), axis=1)
        min_distance = np.min(distances)
        
        # Nếu distance quá lớn thì coi là Unknown
        # Với Facenet: distance < 10 thường là same person
        if min_distance > confidence_threshold:
            return "Unknown", min_distance
        
        return str(pred), min_distance
        
    except Exception as e:
        print(f"Error predicting: {e}")
        return "Unknown", None

def draw_vietnamese_text(img_bgr, text, pos, font_size=24, color=(0, 255, 0)):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype("arial.ttf", font_size)

    draw.text(pos, text, font=font, fill=color)

    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# =========================
# PAGE 1: CHỤP ẢNH
# =========================
def page_chup_anh():
    st.markdown('<div class="main-title">HỆ THỐNG NHẬN DIỆN KHUÔN MẶT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Thu thập dữ liệu với DeepFace RetinaFace Detection</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="step-box">', unsafe_allow_html=True)
    st.subheader("📸 Chụp ảnh thu thập dữ liệu")

    person_name = st.text_input(
        "Nhập tên (có thể có dấu, ví dụ: Mạnh, Khang, Nguyễn Văn A):",
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
        st.warning("Hãy nhập tên trước khi chụp ảnh.")

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

        with st.spinner("Đang detect khuôn mặt..."):
            faces = detect_and_extract_faces(cv2_img)
        
        if not faces:
            st.error("Không tìm thấy khuôn mặt. Hãy chụp lại gần hơn / sáng hơn.")
            return
        
        # Lấy face lớn nhất
        face_bgr, facial_area = max(faces, key=lambda f: f[1]['w'] * f[1]['h'])

        person_folder_path = os.path.join(DATASET_PATH, person_name.strip())
        success, path = _save_image(face_bgr, person_folder_path)

        if success:
            new_count = len([f for f in os.listdir(person_folder_path) 
                           if os.path.isfile(os.path.join(person_folder_path, f))])
            st.success(f"✅ Đã lưu ảnh: {os.path.basename(path)}")
            st.info(f"📊 Tổng số ảnh hiện có: {new_count}/{TARGET_IMAGES_PER_PERSON}")

            st.image(
                cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB),
                caption=f"Khuôn mặt đã lưu ({facial_area['w']}x{facial_area['h']}px)",
                use_container_width=True,
            )
        else:
            st.error(f"Lỗi khi lưu ảnh: {path}")

    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("⚡ Sau khi chụp đủ ảnh, chạy `python train_model.py` để train model.")


# =========================
# PAGE 2: NHẬN DIỆN ẢNH UPLOAD
# =========================
def page_nhan_dien():
    st.markdown('<div class="main-title">NHẬN DIỆN KHUÔN MẶT</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">DeepFace RetinaFace + Facenet Embedding + KNN</div>',
        unsafe_allow_html=True,
    )

    if knn_model is None:
        st.error("⚠️ Chưa load được model KNN.\n\nHãy chạy `python train_model.py` trước.")
        return

    uploaded_img = st.file_uploader("Chọn ảnh để nhận diện", type=["jpg", "jpeg", "png"])

    if uploaded_img is None:
        return

    bytes_data = uploaded_img.read()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

    if cv2_img is None or cv2_img.size == 0:
        st.error("Không đọc được ảnh.")
        return

    # Detect faces
    with st.spinner("Đang phát hiện khuôn mặt..."):
        faces = detect_and_extract_faces(cv2_img)

    if not faces:
        st.warning("Không phát hiện khuôn mặt nào trong ảnh.")
        st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        return

    # Predict cho từng face
    recognized_count = 0
    unknown_count = 0
    
    with st.spinner(f"Đang nhận diện {len(faces)} khuôn mặt..."):
        for face_bgr, facial_area in faces:
            x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
            
            name, distance = predict_name_from_face(face_bgr)
            
            # Chọn màu
            if name == "Unknown":
                box_color = (0, 0, 255)  # Đỏ
                text_color = (0, 0, 255)
                unknown_count += 1
            else:
                box_color = (0, 255, 0)  # Xanh lá
                text_color = (0, 255, 0)
                recognized_count += 1

            draw_rounded_rectangle(
                cv2_img,
                (x, y),
                (x + w, y + h),
                color=box_color,
                thickness=3,
                radius=20,
            )
            
            cv2_img = draw_vietnamese_text(
                cv2_img,
                name,
                (x, y - 35),      
                font_size=28,
                color=text_color
            )

    # Hiển thị kết quả
    if unknown_count == 0:
        st.success(f"✅ Nhận diện thành công {recognized_count} khuôn mặt!")
    else:
        st.info(f"📊 Kết quả: {recognized_count} nhận diện được, {unknown_count} không xác định")
    
    st.image(
        cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB),
        caption="Kết quả nhận diện",
        use_container_width=True,
    )


# =========================
# MAIN
# =========================
def main():
    st.sidebar.title("🎯 Menu")
    st.sidebar.info(f"**Model:** {MODEL_NAME}\n**Detector:** {DETECTOR_BACKEND}")
    
    choice = st.sidebar.radio("Chọn chức năng", ["📸 Chụp ảnh", "🔍 Nhận diện"])

    if choice == "📸 Chụp ảnh":
        page_chup_anh()
    else:
        page_nhan_dien()


if __name__ == "__main__":
    main()
