import streamlit as st
import cv2
import os
from PIL import Image
import numpy as np
try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_ROOT = os.getcwd() 

DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset") 

if not os.path.exists(DATASET_PATH):
    os.makedirs(DATASET_PATH)


def _save_image(frame, person_folder_path):
    """Hàm phụ dùng để lưu ảnh (Hỗ trợ Unicode)"""
    if not os.path.exists(person_folder_path):
        os.makedirs(person_folder_path)
        
    try:
        is_success, img_encoded = cv2.imencode(".jpg", frame)
        if is_success:
            
            count = len(os.listdir(person_folder_path)) + 1
            file_path = os.path.join(person_folder_path, f"{count}.jpg")

            with open(file_path, 'wb') as f:
                f.write(img_encoded.tobytes())
            
            print(f"Đã lưu ảnh: {file_path}") 
            return True, file_path
        else:
            print("Lỗi: cv2.imencode() thất bại.")
            return False, "Lỗi mã hóa ảnh"
    except Exception as e:
        print(f"Lỗi hệ thống khi lưu file: {e}")
        return False, str(e)

st.set_page_config(page_title="Hệ thống Nhận diện", layout="wide")
st.title("HỆ THỐNG NHẬN DIỆN KHUÔN MẶT")

col1, col2 = st.columns([2, 3]) 

with col1:
    st.header("Bước 1: Thu thập Dữ liệu")
    st.write("Nhập tên, sau đó bấm nút chụp ảnh nhiều lần.")
    
    person_name = st.text_input("Nhập tên của bạn:", "TenNguoiMau")
    
    picture = st.camera_input("Chụp ảnh (Thẳng, Nghiêng Trái, Nghiêng Phải)", key="camera_capture")

    if picture is not None:
        if not person_name or person_name == "TenNguoiMau" or person_name.strip() == "":
            st.error("Vui lòng nhập tên của bạn trước khi chụp!")
        else:
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            if cv2_img is not None and cv2_img.size > 0:
                person_folder_path = os.path.join(DATASET_PATH, person_name.strip())
                success, path = _save_image(cv2_img, person_folder_path)
                
                if success:
                    st.success(f"Đã lưu: {os.path.basename(path)}")
                    st.info("Chụp ảnh tiếp theo (góc khác)...")
                else:
                    st.error(f"Lỗi khi lưu: {path}")
            else:
                st.error("Không thể đọc dữ liệu ảnh từ camera.")


with col2:
    st.header("Thông tin Dữ liệu")
    st.write("Kiểm tra số lượng ảnh đã có (Bấm Cập nhật).")

    if st.button("Cập nhật danh sách", key="refresh_sidebar"):
        st.rerun()

    if os.path.exists(DATASET_PATH):
        try:
            
            folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
            if folders:
                st.write(f"Đã có dữ liệu của {len(folders)} người:")
                for folder in folders:
                    try:
                        count = len([f for f in os.listdir(os.path.join(DATASET_PATH, folder)) if os.path.isfile(os.path.join(DATASET_PATH, folder, f))])
                        st.markdown(f"- **{folder}**: {count} ảnh")
                    except Exception as e:
                        st.warning(f"Không thể đọc thư mục: {folder}")
            else:
                st.write("Chưa có dữ liệu.")
        except Exception as e:
            st.error(f"Không thể đọc thư mục dataset: {e}")