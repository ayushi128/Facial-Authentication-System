"""
STEP 4: Live Webcam Authentication App
CO01: Encryption, secure workspace
CO02: Stacking model, positive/negative landmark frames
CO04: Stability buffer, confidence analysis
"""

import cv2
import numpy as np
import pickle
import os
import time
import torch
import torch.nn as nn
from torchvision import models, transforms
import mediapipe as mp
from cryptography.fernet import Fernet
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, scrolledtext

FEAT_DIR   = "./features"
ENROLL_DIR = "./enrolled_users"
NOTES_DIR  = "./workspaces"
os.makedirs(ENROLL_DIR, exist_ok=True)
os.makedirs(NOTES_DIR, exist_ok=True)

# ── LOAD DATA ──────────────────────────────────────────────────────────────────
print("Loading model and splits...")
with open(f"{FEAT_DIR}/splits.pkl", "rb") as f:
    data = pickle.load(f)
with open(f"{FEAT_DIR}/model_config.pkl", "rb") as f:
    cfg = pickle.load(f)

key         = open("./encryption.key", "rb").read()
fernet      = Fernet(key)
le          = data["label_encoder"]
scaler      = data["scaler"]
num_classes = cfg["num_classes"]
label_map   = cfg["label_map"]
device      = "cuda" if torch.cuda.is_available() else "cpu"

# ── BACKBONE ──────────────────────────────────────────────────────────────────
def build_model(n):
    m = models.resnet50(weights=None)
    for p in m.parameters(): p.requires_grad = False
    m.fc = nn.Sequential(
        nn.Linear(2048, 512), nn.BatchNorm1d(512),
        nn.ReLU(), nn.Dropout(0.4), nn.Linear(512, n))
    return m

backbone = build_model(num_classes)
backbone.load_state_dict(torch.load(
    f"{FEAT_DIR}/finetuned_model.pth", map_location=device))
backbone = backbone.to(device)
backbone.eval()
print("✓ Backbone loaded")

# ── STACKING MODEL ────────────────────────────────────────────────────────────
print("Building stacking model...")
X_all = np.vstack([data["X_train"], data["X_test"]])
y_all = np.concatenate([data["y_train"], data["y_test"]])

stacking_model = StackingClassifier(
    estimators=[
        ("svm", SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42)),
        ("knn", KNeighborsClassifier(n_neighbors=5, metric="cosine", n_jobs=-1)),
        ("rf",  RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("mlp", MLPClassifier(hidden_layer_sizes=(256,), max_iter=100, random_state=42)),
        ("nb",  GaussianNB()),
        ("lda", LinearDiscriminantAnalysis()),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=3, n_jobs=-1
)
stacking_model.fit(X_all, y_all)
print(f"✓ Stacking ready | Train Acc: {stacking_model.score(X_all, y_all):.3f}")

# ── EMBEDDING ─────────────────────────────────────────────────────────────────
img_transform = transforms.Compose([
    transforms.ToPILImage(), transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def get_embedding(face_rgb):
    captured = []
    def hook_fn(m, i, o): captured.append(o.detach().cpu().numpy())
    handle = backbone.fc[2].register_forward_hook(hook_fn)
    try:
        t = img_transform(face_rgb).unsqueeze(0).to(device)
        with torch.no_grad(): _ = backbone(t)
        return captured[0][0] if captured else None
    except: return None
    finally: handle.remove()

def get_predicted_name(embedding):
    emb_scaled = scaler.transform(embedding.reshape(1, -1))
    proba      = stacking_model.predict_proba(emb_scaled)[0]
    pred_idx   = np.argmax(proba)
    confidence = proba[pred_idx]
    pred_class = int(stacking_model.classes_[pred_idx])
    name       = label_map.get(pred_class, str(pred_class))
    return name, float(confidence), proba

# ── MEDIAPIPE ─────────────────────────────────────────────────────────────────
mp_face_mesh = mp.solutions.face_mesh
mp_face_det  = mp.solutions.face_detection

# Landmark indices per region
LANDMARK_GROUPS = {
    "L Eye": ([33, 133, 160, 159, 158, 144, 145, 153], (0, 255, 255)),
    "R Eye": ([362, 263, 387, 386, 385, 373, 374, 380], (0, 255, 255)),
    "Nose":  ([1, 4, 19, 94, 168, 5, 195, 197],         (255, 165, 0)),
    "Lips":  ([61, 185, 40, 39, 37, 0, 267, 269, 270,
               409, 291, 146, 91, 181, 84, 17, 314,
               405, 321, 375],                           (255, 50, 150)),
}

# How much each region contributes to face recognition
# Eyes + Nose = high contribution (primary features)
# Lips = lower contribution
REGION_CONTRIBUTION = {
    "L Eye": 0.85,   # high — eyes are primary identifier
    "R Eye": 0.85,   # high
    "Nose":  0.75,   # medium-high
    "Lips":  0.40,   # lower — lips change with expression
}

def draw_landmarks_with_contribution(frame, mesh_results, authenticated, confidence):
    """
    CO02/CO04: Positive/Negative frames based on:
    - Whether currently authenticated
    - Region's inherent contribution to face recognition
    - Landmark visibility in current frame
    
    GREEN (+) = region is contributing to authentication
    RED   (-) = region is not contributing enough
    """
    labeled = frame.copy()
    h, w    = frame.shape[:2]
    contributions = {}

    if not mesh_results.multi_face_landmarks:
        return labeled, contributions

    for face_lm in mesh_results.multi_face_landmarks:
        lm = face_lm.landmark
        for region_name, (indices, color) in LANDMARK_GROUPS.items():
            pts = np.array([
                (int(lm[i].x * w), int(lm[i].y * h)) for i in indices
            ], dtype=np.int32)

            # Visibility of this region in current frame
            visibility = np.mean([
                lm[i].visibility if (hasattr(lm[i], 'visibility') and lm[i].visibility > 0) else 0.9
                for i in indices
            ])

            # Contribution score:
            # region weight × visibility × (1 if authenticated else 0.3)
            auth_factor  = 1.0 if authenticated else 0.3
            contribution = REGION_CONTRIBUTION[region_name] * visibility * auth_factor

            # Positive if authenticated AND landmark is visible
            is_positive = authenticated and visibility > 0.3

            contributions[region_name] = {
                "positive":     is_positive,
                "contribution": contribution,
                "visibility":   visibility,
            }

            # Draw colored dots for region
            for pt in pts:
                cv2.circle(labeled, tuple(pt), 3, color, -1, cv2.LINE_AA)

            # GREEN border = positive, RED border = negative
            if len(pts) > 2:
                border = (0, 255, 0) if is_positive else (0, 0, 255)
                cv2.polylines(labeled, [cv2.convexHull(pts)],
                              True, border, 2, cv2.LINE_AA)

            # Label with +/-
            cx  = int(np.mean(pts[:, 0]))
            cy  = int(np.min(pts[:, 1])) - 8
            ind = "+" if is_positive else "-"
            lc  = (0, 255, 0) if is_positive else (0, 0, 255)
            cv2.putText(labeled, f"{region_name}({ind})",
                        (cx, max(cy, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, lc, 1, cv2.LINE_AA)

    return labeled, contributions

def draw_auth_overlay(frame, identity, confidence, claimed, status, stable=0):
    h, w = frame.shape[:2]
    ov   = frame.copy()
    cv2.rectangle(ov, (0,0), (w,60), (20,20,20), -1)
    cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)

    if status == "authenticated":
        color = (0, 220, 100)
        text  = f"ACCESS GRANTED: {identity}"
    elif status == "mismatch":
        color = (0, 50, 220)
        text  = f"ACCESS DENIED | claimed: {claimed} | detected: {identity}"
    elif status == "unknown":
        color = (0, 50, 220)
        text  = "UNKNOWN FACE"
    else:
        # detecting — show progress
        color = (200, 200, 0)
        if identity.lower() == claimed.lower() and confidence >= 0.02:
            text = f"Verifying {identity}... ({stable}/{5} frames confirmed)"
        else:
            text = f"Scanning... detected: {identity} | Conf: {confidence:.0%}"

    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Conf: {confidence:.1%}", (w-160, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200,200,200), 1, cv2.LINE_AA)
    cv2.putText(frame, "Stacking Ensemble (CO02)",
                (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (150,150,150), 1)
    return frame

def draw_legend(combined, contributions):
    legend = np.zeros((30, combined.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, "Contributions:", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200,200,200), 1)
    x = 140
    for region, d in contributions.items():
        color = (0,255,0) if d.get("positive") else (0,0,255)
        ind   = "+" if d.get("positive") else "-"
        cv2.putText(legend, f"{region}({ind})",
                    (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1)
        x += 110
    cv2.putText(legend, "GREEN(+)=contributing  RED(-)=not contributing",
                (combined.shape[1]-350, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (150,150,150), 1)
    return np.vstack([combined, legend])

# ── WORKSPACE — runs in MAIN THREAD after camera closes ───────────────────────
def open_workspace_blocking(username):
    """
    Opens tkinter workspace in MAIN THREAD (not a daemon thread).
    This keeps the app alive until user closes workspace.
    Saves to txt file + encrypted backup (CO01).
    """
    notes_path = os.path.join(NOTES_DIR, f"{username}_notes.txt")
    enc_path   = os.path.join(NOTES_DIR, f"{username}_notes.enc")

    def load():
        if os.path.exists(notes_path):
            with open(notes_path, "r", encoding="utf-8") as f:
                return f.read()
        return ""

    def save(content):
        with open(notes_path, "w", encoding="utf-8") as f:
            f.write(content)
        with open(enc_path, "wb") as f:
            f.write(fernet.encrypt(content.encode("utf-8")))

    root = tk.Tk()
    root.title(f"🔒 {username}'s Workspace — AES Encrypted (CO01)")
    root.geometry("750x550")
    root.configure(bg="#1e1e1e")

    # Header
    hdr = tk.Frame(root, bg="#2d2d2d", pady=6)
    hdr.pack(fill=tk.X)
    tk.Label(hdr, text=f"🔐 {username}'s Private Workspace",
             font=("Arial",13,"bold"), fg="#00e676", bg="#2d2d2d").pack(side=tk.LEFT, padx=10)
    tk.Label(hdr, text="AES-128 Encrypted | CO01",
             font=("Arial",9), fg="#888", bg="#2d2d2d").pack(side=tk.RIGHT, padx=10)

    # Text area
    tf = tk.Frame(root, bg="#1e1e1e")
    tf.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    txt = scrolledtext.ScrolledText(
        tf, font=("Consolas",11), bg="#252526", fg="#d4d4d4",
        insertbackground="white", selectbackground="#264f78",
        wrap=tk.WORD, relief=tk.FLAT)
    txt.pack(fill=tk.BOTH, expand=True)

    existing = load()
    if existing:
        txt.insert(tk.END, existing)

    # Status bar
    status = tk.StringVar(value=f"✓ Notes loaded from {notes_path}")
    tk.Label(root, textvariable=status, font=("Arial",9),
             fg="#888", bg="#1e1e1e", anchor=tk.W, padx=10).pack(fill=tk.X)

    # Buttons
    bf = tk.Frame(root, bg="#2d2d2d", pady=6)
    bf.pack(fill=tk.X)

    def do_save():
        content = txt.get("1.0", tk.END).rstrip()
        save(content)
        status.set(f"✓ Saved & encrypted — {time.strftime('%H:%M:%S')} | {notes_path}")

    def do_clear():
        if messagebox.askyesno("Clear", "Clear all notes?"):
            txt.delete("1.0", tk.END)
            status.set("Cleared")

    tk.Button(bf, text="💾 Save & Encrypt", command=do_save,
              bg="#0e639c", fg="white", font=("Arial",11),
              relief=tk.FLAT, padx=12, pady=4).pack(side=tk.LEFT, padx=8)
    tk.Button(bf, text="🗑 Clear All", command=do_clear,
              bg="#5a1e1e", fg="white", font=("Arial",11),
              relief=tk.FLAT, padx=12, pady=4).pack(side=tk.LEFT, padx=4)
    tk.Label(bf, text=f"Saved to: {notes_path}",
             font=("Arial",8), fg="#555", bg="#2d2d2d").pack(side=tk.RIGHT, padx=10)

    # Auto-save on close
    def on_close():
        content = txt.get("1.0", tk.END).rstrip()
        if content:
            save(content)
            print(f"✓ Auto-saved notes for {username}")
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()  # BLOCKS until window is closed

# ── LOGIN/REGISTER DIALOG ─────────────────────────────────────────────────────
def show_login_screen():
    result = {"mode": None, "name": None}

    def on_login():
        name = entry.get().strip()
        if name:
            result["mode"] = "login"
            result["name"] = name
            root.destroy()

    def on_register():
        name = entry.get().strip()
        if name:
            result["mode"] = "register"
            result["name"] = name
            root.destroy()

    root = tk.Tk()
    root.title("Facial Authentication")
    root.geometry("420x260")
    root.configure(bg="#1e1e1e")
    root.resizable(False, False)

    tk.Label(root, text="🔐 Facial Authentication System",
             font=("Arial",14,"bold"), fg="#00e676", bg="#1e1e1e").pack(pady=15)
    tk.Label(root, text="Enter your username:",
             font=("Arial",10), fg="#ccc", bg="#1e1e1e").pack()

    entry = tk.Entry(root, font=("Arial",12), width=25,
                     bg="#252526", fg="white", insertbackground="white")
    entry.pack(pady=10)
    entry.focus()

    bf = tk.Frame(root, bg="#1e1e1e")
    bf.pack(pady=5)
    tk.Button(bf, text="🔓 Login", command=on_login,
              bg="#0e639c", fg="white", font=("Arial",11),
              relief=tk.FLAT, padx=20, pady=8, width=16).pack(side=tk.LEFT, padx=8)
    tk.Button(bf, text="📝 Register", command=on_register,
              bg="#2d6a2d", fg="white", font=("Arial",11),
              relief=tk.FLAT, padx=20, pady=8, width=16).pack(side=tk.LEFT, padx=8)

    tk.Label(root, text="Login = verify face | Register = enroll new face",
             font=("Arial",8), fg="#666", bg="#1e1e1e").pack(pady=6)
    entry.bind("<Return>", lambda e: on_login())
    root.mainloop()
    return result["mode"], result["name"]

# ── ENROLLMENT ────────────────────────────────────────────────────────────────
def enroll_new_person(cap, person_name, n_captures=15):
    print(f"\n=== Enrolling: {person_name} ===")
    person_dir  = Path(ENROLL_DIR) / person_name
    person_dir.mkdir(exist_ok=True)
    existing    = list(person_dir.glob("*.jpg"))
    count       = len(existing)
    start_count = count
    target      = start_count + n_captures
    antispoof   = count >= 50
    MAX_FAILS   = 5 if antispoof else 999

    print(f"  Existing: {count} | Adding: {n_captures} | Anti-spoof: {'ON' if antispoof else 'OFF'}")

    fdet = mp_face_det.FaceDetection(min_detection_confidence=0.7)
    fails = 0

    while count < target:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = fdet.process(rgb)
        h, w  = frame.shape[:2]
        face_ok = res.detections is not None and len(res.detections) > 0

        if antispoof and face_ok and count > start_count:
            det  = res.detections[0]
            bbox = det.location_data.relative_bounding_box
            x1 = max(0, int(bbox.xmin*w)-30); y1 = max(0, int(bbox.ymin*h)-30)
            x2 = min(w, int((bbox.xmin+bbox.width)*w)+30)
            y2 = min(h, int((bbox.ymin+bbox.height)*h)+30)
            fc = rgb[y1:y2, x1:x2]
            if fc.size > 0:
                emb = get_embedding(fc)
                if emb is not None:
                    dname, conf, _ = get_predicted_name(emb)
                    if dname.lower() != person_name.lower() and dname != "Unknown" and conf > 0.1:
                        fails += 1
                        if fails >= MAX_FAILS:
                            print(f"🚨 DANGER! Expected {person_name}, got {dname}")
                            alert = frame.copy()
                            cv2.rectangle(alert, (0,0), (w,h), (0,0,255), 8)
                            cv2.putText(alert, "⚠ UNAUTHORIZED — ABORTED",
                                        (w//2-200, h//2), cv2.FONT_HERSHEY_SIMPLEX,
                                        1.0, (0,0,255), 3)
                            cv2.imshow("Enrollment", alert)
                            cv2.waitKey(3000)
                            for i in range(start_count, count):
                                p = person_dir / f"{i:03d}.jpg"
                                if p.exists(): p.unlink()
                            fdet.close(); cv2.destroyWindow("Enrollment"); return
                    else:
                        fails = 0

        color = (0,255,0) if face_ok else (0,0,255)
        cx, cy = w//2, h//2; sz = min(h,w)//2
        cv2.rectangle(frame, (cx-sz//2, cy-sz//2), (cx+sz//2, cy+sz//2), color, 2)
        ov = frame.copy()
        cv2.rectangle(ov, (0,0), (w,55), (20,20,20), -1)
        cv2.addWeighted(ov, 0.7, frame, 0.3, 0, frame)
        cv2.putText(frame, f"Enrolling: {person_name} | {count-start_count}/{n_captures}",
                    (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,200), 2)
        cv2.putText(frame, "SPACE=capture | ESC=cancel",
                    (10,46), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150,150,150), 1)
        cv2.imshow("Enrollment", frame)

        kp = cv2.waitKey(1) & 0xFF
        if kp == 32 and face_ok:
            cv2.imwrite(str(person_dir / f"{count:03d}.jpg"), frame)
            count += 1
            print(f"  Captured {count-start_count}/{n_captures}")
        elif kp == 27:
            break

    fdet.close(); cv2.destroyWindow("Enrollment")
    print(f"✓ Total for {person_name}: {count}")

# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    print("\n=== Facial Authentication System (CO01 + CO02) ===")

    mode, username = show_login_screen()
    if not mode or not username:
        print("Cancelled."); return

    if mode == "register":
        cap_reg = cv2.VideoCapture(0)
        enroll_new_person(cap_reg, username, n_captures=15)
        cap_reg.release(); cv2.destroyAllWindows()
        print("✓ Done! Re-run 01→02→03 then login.")
        return

    print(f"\nAuthenticating as: {username}")
    print("Controls: [Q]=quit [S]=screenshot [L]=landmarks [A]=add photos\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Webcam not found!"); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True,
                                       min_detection_confidence=0.6,
                                       min_tracking_confidence=0.5)
    face_det  = mp_face_det.FaceDetection(min_detection_confidence=0.7)

    show_landmarks = True
    frame_count    = 0
    identity       = "Detecting..."
    confidence     = 0.0
    auth_status    = "detecting"
    contributions  = {}
    stable_count   = 0
    STABLE_NEEDED  = 5

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mesh_results = face_mesh.process(rgb)
        det_results  = face_det.process(rgb)

        if show_landmarks:
            # GREEN when name matches (contributing), RED when not matching
            name_matching = (
                (identity.lower() == username.lower() and confidence >= 0.02)
                or auth_status == "authenticated"
            )
            labeled_frame, contributions = draw_landmarks_with_contribution(
                frame, mesh_results, name_matching, confidence)
        else:
            labeled_frame = frame.copy()
            contributions = {}

        # Authenticate every 10 frames
        if frame_count % 10 == 0 and det_results.detections:
            det  = det_results.detections[0]
            bbox = det.location_data.relative_bounding_box
            h, w = frame.shape[:2]
            x1 = max(0, int(bbox.xmin*w)-30)
            y1 = max(0, int(bbox.ymin*h)-30)
            x2 = min(w, int((bbox.xmin+bbox.width)*w)+30)
            y2 = min(h, int((bbox.ymin+bbox.height)*h)+30)
            fc = rgb[y1:y2, x1:x2]
            if fc.size > 0:
                emb = get_embedding(fc)
                if emb is not None:
                    pred_name, conf, _ = get_predicted_name(emb)
                    confidence = conf
                    identity   = pred_name
                    if conf >= 0.02 and pred_name.lower() == username.lower():
                        # Name matches — increment stable counter
                        stable_count = min(stable_count + 1, STABLE_NEEDED + 5)
                        # Keep showing "detecting" until stable enough
                        if stable_count >= STABLE_NEEDED:
                            auth_status = "authenticated"
                        else:
                            auth_status = "detecting"
                    else:
                        # Name doesn't match — decrement counter slowly
                        stable_count = max(0, stable_count - 1)
                        # Only show denied if we were never authenticated
                        if auth_status != "authenticated":
                            if pred_name == "Unknown" or conf < 0.02:
                                auth_status = "unknown"
                            else:
                                auth_status = "mismatch"

        # Once stable authenticated → close camera → open workspace
        if auth_status == "authenticated":
            # Show granted screen for 1 second
            af = draw_auth_overlay(frame, identity, confidence,
                                   username, "authenticated", stable_count)
            al = draw_auth_overlay(labeled_frame, identity, confidence,
                                   username, "authenticated", stable_count)
            h, w = af.shape[:2]
            scale = 0.6
            fs  = cv2.resize(af, (0,0), fx=scale, fy=scale)
            ls  = cv2.resize(al, (0,0), fx=scale, fy=scale)
            div = np.ones((fs.shape[0], 4, 3), dtype=np.uint8) * 120
            combined = np.hstack([fs, div, ls])
            final    = draw_legend(combined, contributions)
            cv2.imshow(f"Facial Auth — {username}", final)
            cv2.waitKey(1000)

            # Close everything
            cap.release()
            face_mesh.close()
            face_det.close()
            cv2.destroyAllWindows()

            print(f"✓ Authenticated as {username}! Opening workspace...")
            # Open workspace in MAIN THREAD — blocks until closed
            open_workspace_blocking(username)
            print("✓ Workspace closed. Goodbye!")
            return  # exit cleanly

        # Draw overlays
        frame         = draw_auth_overlay(frame, identity, confidence,
                                           username, auth_status, stable_count)
        labeled_frame = draw_auth_overlay(labeled_frame, identity, confidence,
                                           username, auth_status, stable_count)

        h, w = frame.shape[:2]
        cv2.putText(frame,         "Original",       (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
        cv2.putText(labeled_frame, "Landmarks (+/-)", (10, h-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

        scale = 0.6
        fs  = cv2.resize(frame,         (0,0), fx=scale, fy=scale)
        ls  = cv2.resize(labeled_frame, (0,0), fx=scale, fy=scale)
        div = np.ones((fs.shape[0], 4, 3), dtype=np.uint8) * 120
        combined = np.hstack([fs, div, ls])
        final    = draw_legend(combined, contributions)

        cv2.imshow(
            f"Facial Auth — {username} | [Q]quit [A]add photos [S]shot [L]landmarks",
            final)

        frame_count += 1
        kp = cv2.waitKey(1) & 0xFF

        if kp == ord("q"):
            break
        elif kp == ord("s"):
            fname = f"screenshot_{username}_{int(time.time())}.jpg"
            cv2.imwrite(fname, final)
            print(f"✓ Screenshot: {fname}")
        elif kp == ord("l"):
            show_landmarks = not show_landmarks
        elif kp == ord("a"):
            cap.release(); face_mesh.close(); face_det.close()
            cv2.destroyAllWindows()
            cap2 = cv2.VideoCapture(0)
            enroll_new_person(cap2, username, n_captures=15)
            cap2.release()
            cap       = cv2.VideoCapture(0)
            face_mesh = mp_face_mesh.FaceMesh(max_num_faces=2, refine_landmarks=True,
                                               min_detection_confidence=0.6,
                                               min_tracking_confidence=0.5)
            face_det  = mp_face_det.FaceDetection(min_detection_confidence=0.7)

    if cap.isOpened(): cap.release()
    cv2.destroyAllWindows()
    print("\n✓ Session ended.")

if __name__ == "__main__":
    main()
