import cv2
import threading
import time
import queue
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageTk
from test_5 import process
import tkinter as tk

# --------- CONFIG ---------
model = YOLO("yolov8l.pt").to("cuda")
predefined_bbox = (0, 0, 2560, 1440)

frame_queue = queue.Queue(maxsize=10)
processed_queue = queue.Queue(maxsize=10)

stop_event = threading.Event()

# --------- YOLO CROP FUNCTION ---------
def crop_yolo(frame):
    results = model.predict(frame, device="cuda", verbose=False)
    for result in results:
        for box in result.boxes:
            if int(box.cls) in [53, 55] and box.conf >= 0.75:
                xmin, ymin, xmax, ymax = box.xyxy.tolist()[0]
                x1, y1, x2, y2 = predefined_bbox
                if xmin > x1 + 10 and ymin > y1 + 10 and xmax < x2 - 10 and ymax < y2 - 10:
                    return frame[int(ymin):int(ymax), int(xmin):int(xmax)]
    return None

# --------- THREAD 1: CAPTURE ---------
def capture_thread():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera open failed")
        stop_event.set()
        return

    count = 0
    flag = False
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            continue

        x1, y1, x2, y2 = predefined_bbox
        frame_display = frame.copy()
        cv2.rectangle(frame_display, (x1, y1), (x2, y2), (0, 255, 255), 20)

        # Update live view in GUI
        img = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(img).resize((960, 540)))
        live_label.imgtk = imgtk
        live_label.configure(image=imgtk)

        if flag == False:
            crop = crop_yolo(frame)
            count = 0
            if crop is not None and not frame_queue.full():
                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                frame_queue.put(Image.fromarray(crop_rgb))
                flag = True
        else:
            count += 1
            if count % 2 == 0:
                flag = False
                count = 0

        # time.sleep(0.03)

    cap.release()

# --------- THREAD 2: PROCESS ---------
def process_thread():
    i = 0
    while not stop_event.is_set():
        try:
            image = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        processed_image, _ = process(image, i)
        img_cv = cv2.cvtColor(np.array(processed_image), cv2.COLOR_RGB2BGR)
        if not processed_queue.full():
            processed_queue.put(img_cv)
        i += 1

# --------- THREAD 3: DISPLAY ---------
def display_thread():
    while not stop_event.is_set():
        try:
            image = processed_queue.get(timeout=0.1)
        except queue.Empty:
            continue

        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        imgtk = ImageTk.PhotoImage(Image.fromarray(img).resize((960, 540)))
        processed_label.imgtk = imgtk
        processed_label.configure(image=imgtk)

        # Wait for next processed image, don’t clear until then
        while processed_queue.empty() and not stop_event.is_set():
            time.sleep(0.1)

# --------- GUI SETUP ---------
root = tk.Tk()
root.title("CareVision")
root.attributes('-fullscreen', True)
root.configure(bg='black')

frame = tk.Frame(root, bg='black')
frame.pack(fill="both", expand=True)

live_label = tk.Label(frame, bg='black')
live_label.pack(side="left", padx=20, pady=20)

processed_label = tk.Label(frame, bg='black')
processed_label.pack(side="right", padx=20, pady=20)

# Escape key to quit
def on_esc(event):
    stop_event.set()
    root.destroy()

root.bind("<Escape>", on_esc)

# --------- START THREADS ---------
threads = [
    threading.Thread(target=capture_thread, daemon=True),
    threading.Thread(target=process_thread, daemon=True),
    threading.Thread(target=display_thread, daemon=True),
]

for t in threads:
    t.start()

# --------- MAIN LOOP ---------
try:
    root.mainloop()
except KeyboardInterrupt:
    stop_event.set()

stop_event.set()
for t in threads:
    t.join()

print("All threads stopped. Exiting.")
