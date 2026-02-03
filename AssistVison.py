from fastapi import FastAPI, UploadFile, File, Form
import cv2
import numpy as np
import tempfile
import speech_recognition as sr
from ultralytics import YOLO

# ---------------- INIT ----------------
app = FastAPI()

MODEL_PATH = "yolo11n_ncnn_model"
CONF_THRESH = 0.4

model = YOLO(MODEL_PATH)
recognizer = sr.Recognizer()

DANGEROUS_OBJECTS = [
    "knife", "gun", "scissors", "snake", "dog", "bear"
]

# ---------------- HELPERS ----------------
def position_from_bbox(bbox, frame_w):
    x1, _, x2, _ = bbox
    cx = (x1 + x2) / 2
    if cx < frame_w / 3:
        return "बाईं तरफ"
    elif cx > 2 * frame_w / 3:
        return "दाईं तरफ"
    else:
        return "सामने"


def build_one_line_response(detections, ultrasonic):
    if not detections:
        if ultrasonic < 2:
            return f"कोई वस्तु नहीं दिख रही है, लेकिन कोई बहुत पास है, केवल {ultrasonic:.1f} फीट दूर।"
        return "कोई वस्तु नहीं दिख रही है, रास्ता सुरक्षित है।"

    objects_desc = []
    danger = False

    for det in detections[:2]:  # limit to 2 objects
        pos = det["position"]
        name = det["name"]
        objects_desc.append(f"{pos} {name}")
        if name.lower() in DANGEROUS_OBJECTS:
            danger = True

    obj_text = " और ".join(objects_desc)
    dist_text = f"{ultrasonic:.1f} फीट दूर"

    if ultrasonic < 2 or danger:
        warn = "खतरा है, सावधान रहें"
    else:
        warn = "कोई खतरा नहीं"

    return f"{obj_text} है, {dist_text}, {warn}।"


# ---------------- API ----------------
@app.post("/assistvision")
async def assistvision(
    frame: UploadFile = File(...),
    ultrasonic: float = Form(...),
    audio: UploadFile = File(...)
):
    # -------- IMAGE --------
    img_bytes = await frame.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    frame_img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    frame_h, frame_w = frame_img.shape[:2]

    results = model(frame_img, conf=CONF_THRESH, verbose=False)[0]

    detections = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        cls_name = model.names[int(box.cls)]
        pos = position_from_bbox((x1, y1, x2, y2), frame_w)

        detections.append({
            "name": cls_name,
            "bbox": (x1, y1, x2, y2),
            "position": pos
        })

    # -------- AUDIO (STT) --------
    try:
        audio_bytes = await audio.read()
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            f.write(audio_bytes)
            f.flush()
            with sr.AudioFile(f.name) as source:
                audio_data = recognizer.record(source)
                recognizer.recognize_google(audio_data, language="hi-IN")
    except:
        pass  # voice only triggers logic, not required for sentence

    # -------- ONE LINE RESPONSE --------
    response_line = build_one_line_response(detections, ultrasonic)

    return {
        "response": response_line
    }
