#!/usr/bin/env python3
"""
AssistVision – Cloud Server (Render)
Receives:
- ultrasonic distance
- audio bytes
- image frame

Returns:
- Hindi speech audio (WAV bytes)
"""

import io
import time
import cv2
import math
import wave
import numpy as np
import subprocess
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response
from ultralytics import YOLO
from openai import OpenAI

# ---------------- CONFIG ----------------
MODEL_PATH = "models/yolo11n.pt"

DANGEROUS_OBJECTS = [
    "knife", "gun", "scissors", "snake", "stairs"
]

CONF_THRESH = 0.4
HEURISTIC_SPEED_FACTOR = 10.0

# ---------------- INIT ----------------
app = FastAPI()
model = YOLO(MODEL_PATH)

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------- PIPER ----------------
class PiperTTS:
    def __init__(self):
        self.proc = subprocess.Popen(
            [
                "./piper/piper",
                "--model",
                "./piper/models/hi_IN-priyamvada-medium.onnx",
                "--output_file",
                "-"
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

    def speak(self, text: str) -> bytes:
        self.proc.stdin.write(text.encode("utf-8") + b"\n")
        self.proc.stdin.flush()

        audio = b""
        while True:
            chunk = self.proc.stdout.read(4096)
            if not chunk:
                break
            audio += chunk
            if len(chunk) < 4096:
                break
        return audio


tts = PiperTTS()

# ---------------- HELPERS ----------------
def detect_objects(frame):
    results = model(frame, conf=CONF_THRESH, verbose=False)[0]
    dets = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = model.names[int(box.cls)]
        dets.append({
            "name": cls,
            "bbox": (x1, y1, x2, y2)
        })
    return dets


def map_distances(dets, ultrasonic_distance, frame_width):
    mapped = []
    center = frame_width / 2

    for d in dets:
        x1, _, x2, _ = d["bbox"]
        cx = (x1 + x2) / 2
        offset = abs(cx - center) / center
        dist = ultrasonic_distance * (1 + offset * 0.8)

        mapped.append({
            "name": d["name"],
            "distance": round(dist, 1)
        })
    return mapped


def decide_message(dets, distances):
    if not dets:
        return "सामने कोई वस्तु नहीं दिख रही है।"

    warnings = []
    for d in distances:
        if d["name"] in DANGEROUS_OBJECTS and d["distance"] < 100:
            warnings.append(
                f"सावधान! {d['name']} {d['distance']} सेंटीमीटर दूर है।"
            )

    if warnings:
        return " ".join(warnings)

    obj_list = ", ".join(d["name"] for d in dets)
    return f"सामने {obj_list} दिखाई दे रहे हैं।"


def gpt_refine(text, context):
    prompt = f"""
    Context: {context}
    User: {text}
    Reply briefly in Hindi for a blind user.
    """
    try:
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=40
        )
        return r.choices[0].message.content.strip()
    except:
        return context
    
def speech_to_text(audio_bytes: bytes) -> str:
    try:
        # OpenAI expects a file-like object
        audio_file = io.BytesIO(audio_bytes)
        audio_file.name = "audio.wav"  # important!

        transcript = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            language="hi"
        )
        return transcript.text.strip()
    except Exception as e:
        print("STT error:", e)
        return ""


# ---------------- API ----------------
@app.post("/assist")
async def assist(
    ultrasonic_distance: float = Form(...),
    audio: UploadFile = File(...),
    frame: UploadFile = File(...)
):
    # Decode image
    img_bytes = await frame.read()
    img_np = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(img_np, cv2.IMREAD_COLOR)

    # Detect
    detections = detect_objects(image)
    distances = map_distances(detections, ultrasonic_distance, image.shape[1])

    audio_bytes = await audio.read()
    user_text = speech_to_text(audio_bytes)

    if not user_text:
        user_text = "सामने क्या है?"

    # Decide
    base_msg = decide_message(detections, distances)
    final_msg = gpt_refine(user_text, base_msg)

    # TTS
    audio_out = tts.speak(final_msg)

    return Response(
        content=audio_out,
        media_type="audio/wav"
    )
