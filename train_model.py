import os 
import cv2
import numpy as np
import pandas as pd

try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    PROJECT_ROOT = os.getcwd()

DATASET_DIR = os.path.join(PROJECT_ROOT,"dataset")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
CSV_PATH = os.path.join(PROJECT_ROOT, "data.csv")
MODEL_PATH = os.path.join(OUTPUT_DIR, "model_knn.npz")

IMG_SIZE = (100,100)
K = 1


def build_csv():
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Không tìm thấy dataset : {DATASET_DIR}")
    
    features = []
    labels = []

    for person_name in os.listdir(DATASET_DIR):
        person_folder = os.path.join(DATASET_DIR, person_name)
        if not os.path.isdir(person_folder):
            continue

        print(f"Đang đọc ảnh của :{person_name}")
        count_person = 0

        for filename in os.listdir(person_folder):
            if not filename.lower().endswith((".png", "jpg","jpeg")):
                continue

            img_path = os.path.join(person_folder, filename)
            data = np.fromfile(img_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)

            if img is None:
                print("  Không đọc được ảnh:", img_path)
                continue


            img = cv2.resize(img, IMG_SIZE)
            features.append(img.reshape(-1))
            labels.append(person_name)
            count_person += 1
        
        print(f"-> Số ánh dùng được: {count_person}")

    if not features:
        raise ValueError("Không có ảnh hợp lệ trong dataset/")
    
    X = np.array(features, dtype="float32")
    y = np.array(labels)

    df = pd.DataFrame(X, columns=map(str, range(X.shape[1])))
    df["name"] = y
    df.to_csv(CSV_PATH)
    print(f"Đã tạo {CSV_PATH} với {len(df)} dòng.")

    return X, y

def save_model(X, y, k = 5):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.savez(MODEL_PATH, X=X, y=y, k=np.array([k]))

if __name__ == "__main__":
    X, y = build_csv()
    save_model(X, y, K)
