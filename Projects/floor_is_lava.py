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
model = YOLO("../../../Documents/codefolder/pycharmtest/opencv/models/yolo11n-pose.pt")
cap = cv2.VideoCapture("../../../Documents/codefolder/pycharmtest/opencv/assets/pplwalk.mp4")
latest_ankles = []
lock = threading.Lock()

def yolo_thread_func():
    global latest_ankles
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
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

# --- Lava Path Setup ---
ROWS = 10
COLS = 7
STONE_W = WIDTH // (COLS + 1)
STONE_H = HEIGHT // (ROWS + 1)
STONE_SPACING = 30  # pixels between stones
UP_SPEED = 2  # pixels per frame
GAP_ROWS = 2  # max number of rows to skip (force jumps)

def create_safe_path():
    path = []
    col = random.randint(0, COLS - 1)
    row = 0
    while row < ROWS:
        path.append((row, col))
        # Randomly skip 0 or 1 row to create a gap (jump)
        skip = random.choice([1, 1, 2])  # more likely to have 1, sometimes 2 (bigger jump)
        row += skip
        move = random.choice([-1, 0, 1])
        col = max(0, min(COLS - 1, col + move))
    return path

def create_stones(safe_path):
    stones = []
    safe_set = set(safe_path)
    for row in range(ROWS):
        for col in range(COLS):
            if (row, col) in safe_set:
                is_safe = True
            else:
                # Randomly place lava stones, but not on safe path
                is_safe = False if random.random() < 0.7 else None  # None = empty (gap)
            if is_safe is not None:
                x = (col + 1) * STONE_W
                y = HEIGHT - ((row + 1) * STONE_H)
                color = (0, 255, 0) if is_safe else (255, 0, 0)
                stones.append({'rect': pygame.Rect(x, y, STONE_W - STONE_SPACING, STONE_H - STONE_SPACING),
                               'row': row, 'col': col, 'safe': is_safe, 'color': color})
    return stones

safe_path = create_safe_path()
stones = create_stones(safe_path)
offset = 0  # vertical offset for smooth movement

running = True
while running:
    screen.fill((0, 0, 0))

    # Move stones up
    offset += UP_SPEED
    if offset >= STONE_H:
        offset = 0
        # Shift path up, add new safe path at bottom
        safe_path = [(r - 1, c) for r, c in safe_path if r - 1 >= 0]
        # Add new row at bottom
        if safe_path:
            last_row, last_col = safe_path[-1]
        else:
            last_row, last_col = 0, random.randint(0, COLS - 1)
        new_row = ROWS - 1
        move = random.choice([-1, 0, 1])
        new_col = max(0, min(COLS - 1, last_col + move))
        # Randomly skip a row to keep jumps
        if random.random() < 0.5:
            safe_path.append((new_row, new_col))
        stones = create_stones(safe_path)

    # Draw stones
    for stone in stones:
        rect = stone['rect'].copy()
        rect.y += offset
        pygame.draw.rect(screen, stone['color'], rect, 0)
        pygame.draw.rect(screen, (255, 255, 255), rect, 2)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()