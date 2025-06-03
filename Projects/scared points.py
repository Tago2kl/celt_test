
import pygame
import math
import threading
import torch
import cv2
import numpy as np
from ultralytics import YOLO

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
WIDTH, HEIGHT = screen.get_size()
clock = pygame.time.Clock()

# --- YOLO setup ---#
model = YOLO("../../../Documents/codefolder/pycharmtest/opencv/models/yolo11n-pose.pt")
cap = cv2.VideoCapture("../../../Documents/codefolder/pycharmtest/opencv/assets/pplwalk.mp4")
latest_ankles = []
lock = threading.Lock()

def yolo_thread_func():
    global latest_ankles

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        if frame_count % 1 != 0:
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
    def __init__(self, home_x, home_y):
        self.home_x = home_x
        self.home_y = home_y
        self.x = float(home_x)
        self.y = float(home_y)
        self.vx = 0.0
        self.vy = 0.0

    # dont make dampenig over 1  plz
    def update(self, repulse_points, repulse_radius, spring_k=0.0008, damping=0.9, repulse_strength=8.0):
        # diffrence from home
        dx = self.home_x - self.x
        dy = self.home_y - self.y
        #more dis -> faster
        self.vx += dx * spring_k
        self.vy += dy * spring_k

        # repulsion from points
        for rx, ry in repulse_points:
            dist = math.hypot(self.x - rx, self.y - ry)
            if repulse_radius > dist > 1:
                force = repulse_strength * (repulse_radius - dist) / repulse_radius
                self.vx += (self.x - rx) / dist * force
                self.vy += (self.y - ry) / dist * force

        # Damping
        self.vx *= damping
        self.vy *= damping

        # Update position
        self.x += self.vx
        self.y += self.vy

    def draw(self, surface, radius, max_dist=150):
        dist = math.hypot(self.x - self.home_x, self.y - self.home_y)
        t = min(dist / max_dist, 1.0)  # Clamp between 0 and 1

        r = int(255 * (1 - t))
        g = 255
        b = int(255 * (1 - t))

        pygame.draw.circle(surface, (r, g, b), (int(self.x), int(self.y)), radius)

# Grid setup
pointsize = 4
size = [26, 50]  # rows, cols
repulse_radius = 150
grid_points = []
x_spacing = WIDTH // (size[1] + 1)
y_spacing = HEIGHT // (size[0] + 1)

for row in range(1, size[0] + 1):
    for col in range(1, size[1] + 1):
        x = col * x_spacing
        y = row * y_spacing
        grid_points.append(GridPoint(x, y))

while True:
    with lock:
        ankles = latest_ankles.copy()
    repulse_points = [
        (int(ankle[0] * WIDTH / cap.get(3)), int(ankle[1] * HEIGHT / cap.get(4)))
        for left_ankle, right_ankle in ankles
        for ankle in [left_ankle, right_ankle]
    ]
    mouse  = pygame.mouse.get_pos()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    screen.fill((0, 0, 0))
    for point in grid_points:
        point.update(repulse_points, repulse_radius)
        point.update([mouse], repulse_radius)  # Wrap mouse in a list
        point.draw(screen,pointsize)
    pygame.display.flip()
    clock.tick(60)