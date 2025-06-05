import pygame
import math
import threading
import torch
import cv2
import numpy as np
import queue
from ultralytics import YOLO

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
WIDTH, HEIGHT = screen.get_size()
clock = pygame.time.Clock()

# --- YOLO setup ---#
model = YOLO("Detection_Models/yolo11n-pose.pt")
cap = cv2.VideoCapture("Assets/pplwalk.mp4")
latest_ankles = []
lock = threading.Lock()
frame_queue = queue.Queue(maxsize=2)

def yolo_thread_func():
    global latest_ankles
    while True:
        try:
            frame = frame_queue.get(timeout=1)
        except queue.Empty:
            continue
        # Resize frame for faster inference
        small_frame = cv2.resize(frame, (640, 360))
        results = model(small_frame, conf=0.3, stream=True, show=False)
        ankles = []
        for result in results:
            for kpts in result.keypoints.xy:
                if kpts.shape[0] >= 17:
                    # Scale keypoints back to original size
                    scale_x = cap.get(3) / 640
                    scale_y = cap.get(4) / 360
                    left_ankle = (kpts[15].cpu().numpy() * np.array([scale_x, scale_y]))
                    right_ankle = (kpts[16].cpu().numpy() * np.array([scale_x, scale_y]))
                    ankles.append((left_ankle, right_ankle))
        with lock:
            latest_ankles = ankles

# check for gpu availability
if torch.cuda.is_available():
    print(torch.cuda.is_available())
    model.to('cuda')

threading.Thread(target=yolo_thread_func, daemon=True).start()

class GridPoint:
    def __init__(self, home_x, home_y, size):
        self.home_x = home_x
        self.home_y = home_y
        self.x = float(home_x)
        self.y = float(home_y)
        self.vx = 0.0
        self.vy = 0.0
        self.ogsize = size
        self.size = size

    def update(self, repulse_points, repulse_radius, spring_k=0.0008, damping=0.9, repulse_strength=8.0):
        dx = self.home_x - self.x
        dy = self.home_y - self.y
        self.vx += dx * spring_k
        self.vy += dy * spring_k

        for rx, ry in repulse_points:
            dist = math.hypot(self.x - rx, self.y - ry)
            if repulse_radius > dist > 1:
                force = repulse_strength * (repulse_radius - dist) / repulse_radius
                self.vx += (self.x - rx) / dist * force
                self.vy += (self.y - ry) / dist * force

        self.vx *= damping
        self.vy *= damping

        self.x += self.vx
        self.y += self.vy

    def draw(self, surface, radius, max_dist=150, max_size=8):
        dist = math.hypot(self.x - self.home_x, self.y - self.home_y)
        t = min(dist / max_dist, 1.0)
        self.size = int(self.ogsize + (max_size - self.ogsize) * t)
        r = int(255 * (1 - t))
        g = 255
        b = int(255 * (1 - t))
        pygame.draw.circle(surface, (r, g, b), (int(self.x), int(self.y)), self.size)

# Grid setup
pointsize = 4
size = [int(26*1.5), int(50*1.5)]  # rows, cols
repulse_radius = 100
grid_points = []
x_spacing = WIDTH // (size[1] + 1)
y_spacing = HEIGHT // (size[0] + 1)

for row in range(1, size[0] + 1):
    for col in range(1, size[1] + 1):
        x = col * x_spacing
        y = row * y_spacing
        grid_points.append(GridPoint(x, y, pointsize))

last_frame = None

while True:
    ret, frame = cap.read()
    if ret:
        last_frame = frame.copy()
        try:
            frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass  # Drop frame if queue is full

    screen.fill((0, 0, 0))

    with lock:
        ankles = latest_ankles.copy()
    repulse_points = [
        (int(ankle[0] * WIDTH / cap.get(3)), int(ankle[1] * HEIGHT / cap.get(4)))
        for left_ankle, right_ankle in ankles
        for ankle in [left_ankle, right_ankle]
    ]
    mouse = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    for point in grid_points:
        point.update(repulse_points, repulse_radius)
        point.update([mouse], repulse_radius)
        point.draw(screen, pointsize)
    pygame.display.flip()
    clock.tick(29)

cap.release()
pygame.quit()