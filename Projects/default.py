import pygame
import math
import threading
import torch
import cv2
import numpy as np
import random
from ultralytics import YOLO

# --- Initialization ---
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
WIDTH, HEIGHT = screen.get_size()
clock = pygame.time.Clock()

# --- YOLO and Video Setup ---
model = YOLO("Detection_Models/yolo11n-pose.pt")
cap = cv2.VideoCapture("Assets/pplwalk.mp4")
latest_ankles = []
lock = threading.Lock()

def yolo_thread_func():
    global latest_ankles
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        # Example: process frame with YOLO
        small_frame = cv2.resize(frame, (640, 360))
        results = model(small_frame, conf=0.3, stream=True, show=False)
        ankles = []
        for result in results:
            for kpts in result.keypoints.xy:
                if kpts.shape[0] >= 17:
                    scale_x = cap.get(3) / 640
                    scale_y = cap.get(4) / 360
                    left_ankle = (kpts[15].cpu().numpy() * np.array([scale_x, scale_y]))
                    right_ankle = (kpts[16].cpu().numpy() * np.array([scale_x, scale_y]))
                    ankles.append((left_ankle, right_ankle))
        with lock:
            latest_ankles = ankles

if torch.cuda.is_available():
    model.to('cuda')

threading.Thread(target=yolo_thread_func, daemon=True).start()

# --- Main Loop ---
running = True
while running:
    screen.fill((0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Drawing and update logic goes here ---

    pygame.display.flip()
    clock.tick(60)

# --- Cleanup ---
cap.release()
pygame.quit()