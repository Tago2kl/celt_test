import pygame
import math
import threading
import torch
import cv2
import numpy as np
import time
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
latest_keypoints = []
people_count = 0
lock = threading.Lock()
prev_ankles = []

def is_new_person(new_ankle, prev_ankles, threshold=50):
    new_ankle_np = np.array(new_ankle.cpu())
    for prev in prev_ankles:
        prev_np = np.array(prev.cpu())
        if np.linalg.norm(new_ankle_np - prev_np) < threshold:
            return False
    return True

def yolo_thread_func():
    global latest_keypoints, people_count, prev_ankles
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        results = model(frame, conf=0.3, stream=True, show=False)
        new_keypoints = []
        new_ankles = []
        for result in results:
            for kpts in result.keypoints.xy:
                if kpts.shape[0] >= 17:
                    left_ankle = kpts[15]
                    right_ankle = kpts[16]
                    new_keypoints.append((left_ankle, right_ankle))
                    if is_new_person(left_ankle, prev_ankles):
                        with lock:
                            people_count += 1
                    new_ankles.append(left_ankle)
        with lock:
            latest_keypoints = new_keypoints
        prev_ankles = new_ankles

if torch.cuda.is_available():
    model.to('cuda')

threading.Thread(target=yolo_thread_func, daemon=True).start()

# --- Flip Counter Class ---
class FlipCounter:
    def __init__(self, x, y, font, initial=0):
        self.x = x
        self.y = y
        self.font = font
        self.value = initial
        self.display_value = initial
        self.flipping = False
        self.flip_progress = 0.0
        self.flip_speed = 0.15
        self.flip_queue = []

    def set(self, new_value):
        if new_value == self.value:
            return
        direction = 1 if (new_value - self.value) % 10 > 0 else -1
        steps = (new_value - self.value) % 10 if direction == 1 else (self.value - new_value) % 10
        for i in range(steps):
            next_digit = (self.value + direction * (i + 1)) % 10
            self.flip_queue.append(next_digit)
        self.value = new_value

    def update(self):
        if not self.flipping and self.flip_queue:
            self.display_value = self.flip_queue.pop(0)
            self.flipping = True
            self.flip_progress = 0.0
        if self.flipping:
            self.flip_progress += self.flip_speed
            if self.flip_progress >= 1.0:
                self.flipping = False
                self.flip_progress = 0.0

    def draw(self, surface, scale=1.0):
        progress = self.flip_progress if self.flipping else 0
        flip_scale = abs(math.cos(progress * math.pi / 2))
        color = (255, 255, 255)
        digit = str(self.display_value)
        text_surf = self.font.render(digit, True, color)
        w, h = text_surf.get_size()
        final_scale = scale * flip_scale
        scaled_surf = pygame.transform.scale(text_surf, (int(w * scale), int(h * final_scale)))
        rect = scaled_surf.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(scaled_surf, rect)

# --- Main Loop ---
font_path = "font/Modak-Regular.ttf"
font_size = 600
font = pygame.font.Font(font_path, font_size)

def get_digits(n):
    return [int(d) for d in str(n)]

current_count = 0
counters = []
start_time = time.time()

# Shake and zoom effect variables
shake_duration = 0.3  # seconds
shake_timer = 0.0
zoom_scale = 1.0
zoom_target = 1.0
zoom_duration = 0.2  # seconds
zoom_timer = 0.0
last_count = 0

while True:
    screen.fill((0, 0, 0))
    t = time.time() - start_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            exit()
        elif event.type == pygame.MOUSEBUTTONDOWN:
            current_count += 1

    with lock:
        count = people_count
    if count > current_count:
        current_count = count

    # Detect counter increase
    if current_count > last_count:
        shake_timer = shake_duration
        zoom_target = 1.12  # zoom in
        zoom_scale = 1.12
        zoom_timer = zoom_duration
    last_count = current_count

    # Update shake and zoom
    if shake_timer > 0:
        shake_timer -= clock.get_time() / 1000.0
        shake_amount = 12 * (shake_timer / shake_duration)
    else:
        shake_amount = 0

    if zoom_timer > 0:
        zoom_timer -= clock.get_time() / 1000.0
        zoom_scale = 1.0 + (zoom_target - 1.0) * (zoom_timer / zoom_duration)
    else:
        zoom_scale = 1.0

    digits = get_digits(current_count)
    num_digits = len(digits)
    if len(counters) < num_digits:
        for _ in range(num_digits - len(counters)):
            counters.insert(0, FlipCounter(0, HEIGHT // 2, font, 0))
    elif len(counters) > num_digits:
        counters = counters[-num_digits:]

    # Infinity path for the whole group
    a = 60
    speed = 0.25
    inf_x = a * math.sin(speed * t)
    inf_y = a * math.sin(speed * t) * math.cos(speed * t)
    wobble = 12 * math.sin(2 * speed * t)
    seesaw_amp = 50
    seesaw = seesaw_amp * math.sin(speed * t * 1.5)
    total_width = num_digits * font_size * 0.7 * zoom_scale
    start_x = WIDTH // 2 - total_width // 2 + font_size * 0.35 * zoom_scale

    for i, digit in enumerate(digits):
        counters[i].set(digit)
        counters[i].update()
        seesaw_offset = seesaw * (1 - i / max(num_digits - 1, 1))
        # Shake: random offset per digit
        shake_x = random.uniform(-shake_amount, shake_amount) if shake_amount > 0 else 0
        shake_y = random.uniform(-shake_amount, shake_amount) if shake_amount > 0 else 0
        x = int(start_x + i * font_size * 0.7 * zoom_scale + inf_x + wobble + shake_x)
        y = int(HEIGHT // 2 + inf_y + seesaw_offset + shake_y)
        counters[i].x = x
        counters[i].y = y
        counters[i].draw(screen, scale=zoom_scale)

    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()