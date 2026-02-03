#!/usr/bin/env python3
"""
AssistVision (Raspberry Pi Version)
- Object detection (YOLOv8n) + Motion Awareness
- Voice Q/A using SpeechRecognition (Google STT) and Piper (offline TTS)
- Smart Proximity Alert (ultrasonic distance only)
- USB webcam input
- Audio input/output through Bluetooth mic/speaker
"""

import io
import cv2
import pygame
import time
import threading
import subprocess
import speech_recognition as sr
import lgpio
from ultralytics import YOLO
import math
from queue import Queue
from openai import OpenAI

# ---------------- CONFIG ----------------
# Motion tracking globals
prev_positions = {}  # object center → Detection object
prev_times = {}      # object center → last seen timestamp
TRACKING_PIXEL_THRESHOLD = 25  # Max pixel movement between frames for same object
HEURISTIC_SPEED_FACTOR = 10.0 # Multiplier for pixel speed when calibration is absent
# Distance calibration globals
calibration_factor = None
calibration_timestamp = None

CONF_THRESH = 0.4
IOU_MOVING_THRESH = 0.5
MODEL_PATH = "/home/pi/py/yolo11n_ncnn_model"
ENABLE_PREVIEW = False
PROXIMITY_THRESHOLD = 2  # cm
ALERT_COOLDOWN = 3
last_alert_time = 0

TRIG_PIN = 23
ECHO_PIN = 24

try:
    chip = lgpio.gpiochip_open(0)
    lgpio.gpio_claim_output(chip, TRIG_PIN)
    lgpio.gpio_claim_input(chip, ECHO_PIN)
except Exception as e:
    chip = None
    print("lgpio init error:", e)


DANGEROUS_OBJECTS = [
    "knife", "gun", "scissors", "lion", "tiger", "bear", "snake", "stairs", "open_manhole"
]

# ---------------- Helper Functions ----------------
def estimate_distance_ultrasonic():
    """Return distance in feet (fallback-safe)."""
    if chip is None:
        return 999
    try:
        lgpio.gpio_write(chip, TRIG_PIN, 0)
        time.sleep(0.000002)
        lgpio.gpio_write(chip, TRIG_PIN, 1)
        time.sleep(0.00001)
        lgpio.gpio_write(chip, TRIG_PIN, 0)

        t0 = time.time()
        while lgpio.gpio_read(chip, ECHO_PIN) == 0:
            t0 = time.time()
        while lgpio.gpio_read(chip, ECHO_PIN) == 1:
            t1 = time.time()

        pulse = t1 - t0
        cm = pulse * 17150
        feet = round(cm / 30.48, 2)
        #print("distance:", feet, "ft")
        return feet
    except Exception as e:
        print("Distance read error:", e)
        return 999

def get_position(xyxy, frame_width, frame_height):
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    horiz = "बाईं तरफ" if cx < frame_width / 3 else "दाईं तरफ" if cx > 2 * frame_width / 3 else "बीच में"
    vert = "ऊपर" if cy < frame_height / 3 else "नीचे" if cy > 2 * frame_height / 3 else "मध्य में"
    return f"{horiz}, {vert}"

def update_calibration(dets, ultrasonic_distance, frame_width, frame_height):
    # Helper is unchanged
    global calibration_factor, calibration_timestamp

    if ultrasonic_distance > 200 or ultrasonic_distance < 5 or not dets:
        return

    frame_center_x = frame_width / 2
    centered_det = min(dets, key=lambda d: abs(((d.bbox[0] + d.bbox[2]) / 2) - frame_center_x))
    x1, y1, x2, y2 = centered_det.bbox
    pixel_height = abs(y2 - y1)
    
    if pixel_height <= 0:
        return

    new_factor = ultrasonic_distance * pixel_height 
    
    if calibration_factor is None:
        calibration_factor = new_factor
    else:
        calibration_factor = 0.9 * calibration_factor + 0.1 * new_factor

    calibration_timestamp = time.time()

def detect_motion_and_speed(current_positions):
    # Function is unchanged from the last corrected version
    global prev_positions, prev_times, calibration_factor
    now = time.time()
    motions = []
    
    keys_to_delete_on_track = set() 

    for (cx, cy, cls_name), det in current_positions.items():
        prev_key = None
        
        for (pcx, pcy, pcls) in list(prev_positions.keys()):
            if pcls == cls_name and \
               abs(pcx - cx) < TRACKING_PIXEL_THRESHOLD and \
               abs(pcy - cy) < TRACKING_PIXEL_THRESHOLD:
                prev_key = (pcx, pcy, pcls)
                break

        if prev_key and prev_key in prev_times:
            dt = now - prev_times[prev_key]
            if dt > 0:
                dx = cx - prev_key[0]
                dy = cy - prev_key[1]
                pixel_speed = math.sqrt(dx ** 2 + dy ** 2) / dt
                
                real_speed = pixel_speed 
                if calibration_factor:
                    x1, y1, x2, y2 = det.bbox
                    pixel_height = abs(y2 - y1)
                    if pixel_height > 0:
                        D = calibration_factor / pixel_height
                        scale_factor_cm_per_pixel = D * (1 / 1000.0) 
                        real_speed = pixel_speed * scale_factor_cm_per_pixel
                    else:
                        real_speed = pixel_speed * HEURISTIC_SPEED_FACTOR 
                else:
                    real_speed = pixel_speed * HEURISTIC_SPEED_FACTOR 
                
                horiz = "बाईं तरफ" if dx < 0 else "दाईं तरफ" if dx > 0 else "सीधा"
                vert = "ऊपर" if dy < 0 else "नीचे" if dy > 0 else "मध्य"
                motions.append({
                    "object": det.cls_name,
                    "speed": round(real_speed, 1),
                    "direction": f"{horiz}, {vert}"
                })
                
                keys_to_delete_on_track.add(prev_key)

        current_key = (cx, cy, det.cls_name)
        prev_positions[current_key] = det
        prev_times[current_key] = now

    for k in keys_to_delete_on_track:
        if k in prev_positions:
            prev_positions.pop(k, None)
            prev_times.pop(k, None)
            
    keys_to_delete_timeout = [k for k, t in list(prev_times.items()) if now - t > 1.0]
    for k in keys_to_delete_timeout:
        prev_positions.pop(k, None)
        prev_times.pop(k, None)
    return motions

# Local Gemma (Llama.cpp backend)
class OpenAIModel:
    def __init__(self):
        self.client = OpenAI(api_key="",timeout=30)
        self.model = "gpt-4o-mini"  # Fast and small, great for Pi

    def generate(self, prompt):
        try:
            print("request start")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant for a blind user, reply in Hindi."},
                    {"role": "user", "content": prompt}
                    
                ],
                temperature=0.6,
                max_tokens=30
            )
            text = response.choices[0].message.content.strip()
            print("OpenAI output:", text)
            return text
        except Exception as e:
            print("OpenAI error:", e)
            return "no"

# Initialize local Llama
openai_model = OpenAIModel()

def interpret_command(text, detections, frame_width, frame_height, ultrasonic_distance=None):
    print("Inside ic")
    det_info = []

    # Calculate approximate distance for each object using the same logic as map_distances_to_objects()
    object_dists = map_distances_to_objects(detections, ultrasonic_distance or 50, frame_width)

    # Convert list → dict for quick lookup
    dist_lookup = {obj["name"]: obj["distance"] for obj in object_dists}

    for d in detections:
        x1, y1, x2, y2 = d.bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        horiz = "बाईं तरफ" if cx < frame_width / 3 else "दाईं तरफ" if cx > 2 * frame_width / 3 else "बीच में"
        vert = "ऊपर" if cy < frame_height / 3 else "नीचे" if cy > 2 * frame_height / 3 else "मध्य में"

        # Find estimated distance for this object
        distance_val = dist_lookup.get(d.cls_name, ultrasonic_distance or 999)
        distance_info = f"{distance_val:.1f} फीट दूर"

        det_info.append(f"{d.cls_name} {horiz}, {vert}, {distance_info}")

    # अब normal GPT context
    prompt = f"""
    Objects in view: {', '.join(det_info)}.
    User question: "{text}"
    Give a brief and natural answer.
    Answer:"""
    print("generate")
    response = openai_model.generate(prompt)
    if response=="no":
        print("Ai error:")
        if "कौन" in text or "क्या" in text:
            if not det_info:
                return "यहाँ कोई वस्तु नहीं दिख रही है।"
            return "मैंने " + ", ".join(d.cls_name for d in detections) + " देखा।"
        if "दूरी" in text or "कितनी" in text:
            if isinstance(ultrasonic_distance, (int, float)):
                return f"अल्ट्रासोनिक दूरी {ultrasonic_distance:.1f} सेंटीमीटर है।"
            return "अल्ट्रासोनिक दूरी अज्ञात है।"
        if not detections:
            return "कोई वस्तु नहीं दिख रही है।"
        else:
            return "सामने " + ", ".join(det_info) + " दिखाई दे रही हैं।"
    else:
        return response
   

# ---------------- Classes ----------------
class Detection:
    def __init__(self, cls_name, conf, bbox):
        self.cls_name = cls_name
        self.conf = conf
        self.bbox = bbox
class PersistentPiper:
    def __init__(self, piper_path, model_path):
        self.piper_path = piper_path
        self.model_path = model_path

        # Start Piper process only once
        self.process = subprocess.Popen(
            [self.piper_path, "--model", self.model_path, "--output_file", "-", "--sentence_silence", "0.5"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,  # Optional: ignore logs
            bufsize=0
        )
            
            # Pre-warm the model
        self._warmup()

    def _warmup(self):
        try:
            # Synthesize a short text to warm up the model
            self.synthesize("नमस्ते")
        except Exception as e:
            print("Piper warmup error:", e)

    def synthesize(self, text):
        try:
            if not text.strip():
                return None
            # Send text to Piper's stdin
            self.process.stdin.write(text.encode("utf-8") + b"\n")
            self.process.stdin.flush()

            # Piper sends raw audio to stdout — collect it
            audio_bytes = b""
            while True:
                chunk = self.process.stdout.read(4096)
                if not chunk:
                    break
                audio_bytes += chunk
                if len(chunk) < 4096:
                    break
            return audio_bytes
        except Exception as e:
            print("[PersistentPiper Error]:", e)
            return None

    def close(self):
        try:
            self.process.terminate()
        except:
            pass


class Speaker:
    def __init__(self,stt=None):
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=4096)
        self.piper = PersistentPiper(
            "/home/pi/py/piper/piper",
            "/home/pi/py/models/hi_IN-priyamvada-medium.onnx"
        )
        self.lock = threading.Lock()
        self.is_speaking = False 
        self.stt = stt  # Link to STT for pausing/resuming

    def say(self, text):
        if self.is_speaking:  # ✅ ignore if currently speaking
            print("[SAY] Ignored — busy speaking")
            return
        threading.Thread(target=self._do_say, args=(text, False), daemon=True).start()

    def say_alert(self, text):
        if self.is_speaking:  # ✅ skip if busy
            print("[ALERT] Ignored — busy speaking")
            return
        threading.Thread(target=self._do_say, args=(text, True), daemon=True).start()


    def _do_say(self, text, is_alert):
        with self.lock:
            if is_alert:
                pygame.mixer.stop()  # stop ongoing normal speech
                print("[PRIORITY ALERT]:", text)

            self.is_speaking = True
            audio_data = self.piper.synthesize(text)
            
            if audio_data:
                sound = pygame.mixer.Sound(file=io.BytesIO(audio_data))
                sound.play()

                # Alerts wait until fully spoken, but avoid infinite wait
                start_time = time.time()
                TIMEOUT = 15.0  # seconds safety limit

                while pygame.mixer.get_busy():
                    time.sleep(0.01)
                    if time.time() - start_time > TIMEOUT:
                        print("[Speaker Warning] Audio timeout break")
                        break

            time.sleep(0.5)  # brief pause after speaking       
            self.is_speaking = False


    def stop(self):
        pygame.mixer.stop()
        self.piper.close()


class SpeechToText:
    def __init__(self, detector, speaker):
        self.detector = detector
        self.speaker = speaker
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.running = True
        self.lock = threading.Lock()
        self.current_processing = False
        self.last_callback_time = time.time()
        self.recognizer.energy_threshold = 2500  # higher = less sensitive
        self.recognizer.dynamic_energy_threshold = False
        self.yes=True
        self.stop_listening = None

    def callback(self, recognizer, audio):
        if self.speaker and self.speaker.is_speaking:
            print("[STT] Ignored because speaker is talking.")
            return

        self.last_callback_time = time.time()
        if self.current_processing:
            print("Ignoring input, busy processing previous command.")
            return

        def process():
            if self.yes:
                with self.lock:
                    self.current_processing = True
                try:
                    text = recognizer.recognize_google(audio, language="hi-IN")
                    print(f"Command: {text}")
                    dets = self.detector.latest_dets
                    fw, fh = self.detector.latest_frame_shape
                    ultra_dist = estimate_distance_ultrasonic()
                    answer = interpret_command(text, dets, fw, fh, ultra_dist)
                    print(f"Answer: {answer}")
                    answer = answer.replace("**", "").replace("*", "").strip()
                    for i in answer.split("|"):
                        self.speaker.say(i)
                except sr.UnknownValueError:
                    pass
                except Exception as e:
                    print("STT error:", e)
                    self.speaker.say("कृपया अपना नेटवर्क कनेक्शन जाँचें")
                finally:
                    with self.lock:
                        self.current_processing = False

        threading.Thread(target=process, daemon=True).start()

    def start_listening(self):
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source)
        self.stop_listening = self.recognizer.listen_in_background(self.microphone, self.callback)
        print("Started background listening...")

        # 🧭 Start watchdog
        threading.Thread(target=self._watchdog, daemon=True).start()

    def _watchdog(self):
        """Restart listener if no audio received for >30s."""
        while self.running:
            if time.time() - self.last_callback_time > 20:
                self.speaker.say_alert("सुनने की सेवा दोबारा शुरू कर रहा हूँ।")
                print("[STT] Watchdog restarting listener...")
                try:
                    if self.stop_listening:
                        self.stop_listening(wait_for_stop=False)
                    self.stop_listening = self.recognizer.listen_in_background(self.microphone, self.callback)
                    self.last_callback_time = time.time()
                except Exception as e:
                    print("[STT] Watchdog restart failed:", e)
                finally:
                    self.speaker.say("मैं फिर से सुन रहा हूँ।")
            time.sleep(5)

    def pause_listening(self):
        if self.stop_listening:
            try:
                self.stop_listening(wait_for_stop=False)
                print("[STT] Paused listening during speech.")
            except Exception as e:
                print("STT pause error:", e)
            self.stop_listening = None

    def resume_listening(self):
        try:
            self.stop_listening = self.recognizer.listen_in_background(self.microphone, self.callback)
            self.last_callback_time = time.time()
            print("[STT] Resumed listening after speech.")
        except Exception as e:
            print("STT resume error:", e)

    def stop(self):
        self.running = False
        if self.stop_listening:
            self.stop_listening(wait_for_stop=False)



class ThreadedDetector:
    def __init__(self):
        self.detector = Detector()
        self.frame_queue = Queue(maxsize=1)
        self.result_queue = Queue(maxsize=1)
        self.running = True
        self.thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.thread.start()

    def _detection_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get()
                dets, current_positions = self.detector.detect(frame)
                # Clear queue before putting new results
                while not self.result_queue.empty():
                    self.result_queue.get()
                self.result_queue.put((dets, current_positions))
            except Exception as e:
                print(f"Detection thread error: {e}")

    def detect_async(self, frame):
        # Clear queue before putting new frame
        while not self.frame_queue.empty():
            self.frame_queue.get()
        self.frame_queue.put(frame)

    def get_results(self):
        try:
            return self.result_queue.get_nowait()
        except:
            return None

    def stop(self):
        self.running = False


class Detector:
    def __init__(self):
        self.model = YOLO(MODEL_PATH)
        self.prev_dets = []
        self.latest_dets = []
        self.latest_frame_shape = (640, 480)  # Default values

    def detect(self, frame):
        results = self.model(frame, conf=CONF_THRESH, verbose=False)[0]
        dets = []
        current_positions = {}  

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            conf = float(box.conf)
            cls_name = self.model.names[int(box.cls)]
            det = Detection(cls_name, conf, (x1, y1, x2, y2))
            dets.append(det)

            # Center point
            cx, cy = (x1 + x2)/2, (y1 + y2)/2
            current_positions[(cx, cy, det.cls_name)] = det
 # motion tracking

        self.latest_dets = dets
        self.latest_frame_shape = (frame.shape[1], frame.shape[0])
        return dets, current_positions  
    
def map_distances_to_objects(dets, ultrasonic_distance, frame_width):
    # Helper is unchanged
    global calibration_factor
    mapped = []
    frame_center_x = frame_width / 2

    if not dets:
        return mapped

    for det in dets:
        x1, y1, x2, y2 = det.bbox
        cx = (x1 + x2) / 2.0
        pixel_height = abs(y2 - y1)

        if calibration_factor and pixel_height > 0:
            distance_cm = calibration_factor / pixel_height
        else:
            offset_ratio = abs(cx - frame_center_x) / frame_center_x
            distance_cm = ultrasonic_distance * (1 + offset_ratio * 0.8)

        mapped.append({
            "name": det.cls_name,
            "distance": round(distance_cm, 1)
        })
    return mapped


# ---------------- MAIN ----------------
def main():
    global last_alert_time
    threaded_detector = ThreadedDetector()
    detector = threaded_detector.detector  # For STT to access latest_dets
    speaker = Speaker()
    stt = SpeechToText(detector, speaker)
    stt.start_listening()
    speaker.stt = stt  # Link STT to Speaker for pausing/resuming

    # Give a brief welcome message
    speaker.say("आपका स्वागत है, अब आप बात करना शुरू कर सकते हैं") 
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        speaker.say_alert("कैमरा नहीं मिल रहा है।")
        return

    print("Starting AssistVision with real distance calibration...")

    try:
        while True:
            alert_msg = None
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            small_frame = cv2.resize(frame, (224, 224))
            
            # Submit frame for async detection
            threaded_detector.detect_async(small_frame)
            
            ultra_dist = estimate_distance_ultrasonic()
            #print(f"Ultrasonic Distance: {ultra_dist:.1f} feet")
            # Get results if available
            detection_result = threaded_detector.get_results()
            if detection_result:
                dets, current_positions = detection_result
                
                # 3. Calibration Update
                update_calibration(dets, ultra_dist, small_frame.shape[1], small_frame.shape[0])
                
                # 4. Data Processing 
                detector.latest_motions = detect_motion_and_speed(current_positions)
                detector.latest_object_dists = map_distances_to_objects(
                    dets, ultra_dist, small_frame.shape[1]
                )
                
                motions = detector.latest_motions
                object_dists = detector.latest_object_dists


                # 5b. Dangerous moving object within 100 cm
                now = time.time()
                for m in motions:
                    try:
                        if m['object'].lower() in DANGEROUS_OBJECTS and m['speed'] > 30:
                            match = next((o for o in object_dists if o['name'].lower() == m['object'].lower()), None)
                            if match and match['distance'] < 100 and (now - last_alert_time > ALERT_COOLDOWN):
                                # Decide guidance based on fewer objects left vs right
                                frame_w = small_frame.shape[1]
                                left_count = 0
                                right_count = 0
                                for d in dets:
                                    cx = (d.bbox[0] + d.bbox[2]) / 2.0
                                    if cx < frame_w / 3:
                                        left_count += 1
                                    elif cx > 2 * frame_w / 3:
                                        right_count += 1
                                if left_count < right_count:
                                    guidance = "बाएं जाएँ — वहां कम बाधाएँ हैं।"
                                elif right_count < left_count:
                                    guidance = "दाएं जाएँ — वहां कम बाधाएँ हैं।"
                                else:
                                    guidance = "बीच या सावधानीपूर्वक बाएँ/दाएँ जाएँ, दोनों तरफ समान बाधाएँ हैं।"

                                last_alert_time = now
                                alert_msg_d = (f"⚠️ सावधान! {m['object']} पास आ रहा है — दूरी {match['distance']:.1f} फीट, "
                                               f"गति {m['speed']:.1f} feet/s. सुझाव: {guidance}")
                                print("[DANGEROUS MOVING ALERT]", alert_msg_d)
                                speaker.say_alert(alert_msg_d)
                    except Exception:
                        pass

                
                # --- Proximity Alert Logic ---
            now = time.time()
            if ultra_dist< PROXIMITY_THRESHOLD and (now - last_alert_time > ALERT_COOLDOWN):
                alert_msg = f"कोई बहुत करीब है, केवल {ultra_dist:.1f} फीट दूर।"
                last_alert_time = now 
                speaker.say_alert(alert_msg)
                

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nExiting safely...")

    finally:
        threaded_detector.stop()
        stt.stop()
        speaker.stop()
        cap.release()
        cv2.destroyAllWindows()
        


if __name__ == "__main__":
    main()