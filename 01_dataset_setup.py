"""
STEP 1: Dataset Setup & Preprocessing Pipeline
CO01: Pre-processing techniques for ML applications

Uses only: MediaPipe + OpenCV + cryptography (no TensorFlow, no dlib, no MTCNN)
"""

import os
import cv2
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from cryptography.fernet import Fernet
import mediapipe as mp

# ── CONFIG — UPDATE THIS PATH TO YOUR VGGFACE2 FOLDER ─────────────────────────
DATA_ROOT      = r"C:\Ayushi\Amrita\Second_year\Sem-4\Machine_Learning\Project\DATASET\archive\train"
OUTPUT_DIR     = "./processed"
MAX_IDENTITIES = 100
MAX_PER_ID     = 50
IMG_SIZE       = (224, 224)
# ───────────────────────────────────────────────────────────────────────────────

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── ENCRYPTION (CO01) ──────────────────────────────────────────────────────────
def generate_and_save_key(key_path="./encryption.key"):
    key = Fernet.generate_key()
    with open(key_path, "wb") as f:
        f.write(key)
    print(f"[KEY] Saved to {key_path}")
    return key

def encrypt_embedding(embedding, fernet):
    return fernet.encrypt(pickle.dumps(embedding))

def decrypt_embedding(encrypted, fernet):
    return pickle.loads(fernet.decrypt(encrypted))

# ── FACE DETECTION via MediaPipe (CO01) ───────────────────────────────────────
mp_face_det  = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh

def detect_and_crop(img_path, face_detector):
    """CO01: Detect face → crop → resize 224x224 → normalize [0,1]"""
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_detector.process(img_rgb)
    if not results.detections:
        return None

    det  = results.detections[0]
    bbox = det.location_data.relative_bounding_box
    h, w = img_rgb.shape[:2]

    margin = 0.2
    x1 = max(0, int((bbox.xmin - margin * bbox.width)  * w))
    y1 = max(0, int((bbox.ymin - margin * bbox.height) * h))
    x2 = min(w, int((bbox.xmin + bbox.width  * (1 + margin)) * w))
    y2 = min(h, int((bbox.ymin + bbox.height * (1 + margin)) * h))

    if x2 <= x1 or y2 <= y1:
        return None

    face = img_rgb[y1:y2, x1:x2]
    face = cv2.resize(face, IMG_SIZE)
    return face.astype(np.float32) / 255.0

# ── LANDMARK DRAWING (sir's requirement: label eyes/nose/lips) ────────────────
LANDMARK_GROUPS = {
    "L Eye": ([33, 133, 160, 159, 158, 144, 145, 153], (0, 255, 255)),
    "R Eye": ([362, 263, 387, 386, 385, 373, 374, 380], (0, 255, 255)),
    "Nose":  ([1, 4, 19, 94, 168, 5, 195, 197],         (255, 165, 0)),
    "Lips":  ([61, 185, 40, 39, 37, 0, 267, 269, 270,
               409, 291, 146, 91, 181, 84, 17, 314,
               405, 321, 375],                           (255, 50, 150)),
}

def draw_landmarks_sidebyside(img_rgb):
    """Returns [original | labeled] side-by-side as required by sir."""
    h, w = img_rgb.shape[:2]
    labeled = img_rgb.copy()
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                refine_landmarks=True,
                                min_detection_confidence=0.5) as mesh:
        results = mesh.process(img_rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            for name, (indices, color) in LANDMARK_GROUPS.items():
                pts = np.array([(int(lm[i].x * w), int(lm[i].y * h))
                                for i in indices], dtype=np.int32)
                for pt in pts:
                    cv2.circle(labeled, tuple(pt), 3, color, -1, cv2.LINE_AA)
                if len(pts) > 2:
                    cv2.polylines(labeled, [cv2.convexHull(pts)],
                                  True, color, 1, cv2.LINE_AA)
                cx = int(np.mean(pts[:, 0]))
                cy = int(np.min(pts[:, 1])) - 8
                cv2.putText(labeled, name, (cx, max(cy, 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
    divider = np.ones((h, 3, 3), dtype=np.uint8) * 180
    return np.hstack([img_rgb, divider, labeled])

# ── MAIN DATASET BUILDER ───────────────────────────────────────────────────────
def build_dataset():
    print("\n=== Building VGGFace2 Pre-processed Dataset (CO01) ===\n")
    data_path = Path(DATA_ROOT)
    if not data_path.exists():
        print(f"ERROR: Path not found: {DATA_ROOT}")
        print("Update DATA_ROOT at the top of this file to your VGGFace2 train folder.")
        return None, None, None

    identity_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])[:MAX_IDENTITIES]
    print(f"Found {len(identity_dirs)} identity folders. Processing...")

    all_images, all_labels, label_map = [], [], {}

    with mp_face_det.FaceDetection(min_detection_confidence=0.7) as detector:
        for idx, id_dir in enumerate(tqdm(identity_dirs, desc="Identities")):
            label_map[idx] = id_dir.name
            for img_path in list(id_dir.glob("*.jpg"))[:MAX_PER_ID]:
                face = detect_and_crop(img_path, detector)
                if face is not None:
                    all_images.append(face)
                    all_labels.append(idx)

    all_images = np.array(all_images, dtype=np.float32)
    all_labels = np.array(all_labels, dtype=np.int32)

    print(f"\n✓ {len(all_images)} valid faces from {len(label_map)} identities")
    np.save(f"{OUTPUT_DIR}/images.npy", all_images)
    np.save(f"{OUTPUT_DIR}/labels.npy", all_labels)
    with open(f"{OUTPUT_DIR}/label_map.pkl", "wb") as f:
        pickle.dump(label_map, f)
    print(f"✓ Saved to {OUTPUT_DIR}/")
    return all_images, all_labels, label_map

def add_enrolled_users(all_images, all_labels, label_map):
    """Automatically picks up photos from ./enrolled_users/ and adds them."""
    enroll_path = Path("./enrolled_users")
    if not enroll_path.exists():
        return all_images, all_labels, label_map
    
    enrolled_dirs = [d for d in enroll_path.iterdir() if d.is_dir()]
    if not enrolled_dirs:
        print("No enrolled users found, skipping.")
        return all_images, all_labels, label_map
    
    print(f"\nFound {len(enrolled_dirs)} enrolled users. Adding them...")
    next_label = max(label_map.keys()) + 1

    with mp_face_det.FaceDetection(min_detection_confidence=0.7) as detector:
        for id_dir in enrolled_dirs:
            label_map[next_label] = id_dir.name
            img_paths = list(id_dir.glob("*.jpg"))
            print(f"  {id_dir.name}: {len(img_paths)} photos")
            for img_path in img_paths:
                face = detect_and_crop(img_path, detector)
                if face is not None:
                    all_images = np.vstack([all_images, face[np.newaxis]])
                    all_labels = np.append(all_labels, next_label)
            next_label += 1

    return all_images, all_labels, label_map

if __name__ == "__main__":
    if not os.path.exists("./encryption.key"):
        generate_and_save_key()
    else:
        print("[KEY] encryption.key already exists.")

    images, labels, label_map = build_dataset()
    if images is not None:
        images, labels, label_map = add_enrolled_users(images, labels, label_map)
        np.save(f"{OUTPUT_DIR}/images.npy", images)
        np.save(f"{OUTPUT_DIR}/labels.npy", labels)
        with open(f"{OUTPUT_DIR}/label_map.pkl", "wb") as f:
            pickle.dump(label_map, f)
        print(f"\nDone! Total images: {len(images)}, Total identities: {len(label_map)}")
        print("Next: run 02_feature_extraction.py")
