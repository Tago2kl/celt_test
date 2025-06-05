import pygame
import math
import threading
import torch
import cv2
import numpy as np
import random
from ultralytics import YOLO

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
WIDTH, HEIGHT = screen.get_size()
clock = pygame.time.Clock()
tarzan_img = pygame.image.load("Assets/minitarzan.png").convert_alpha()
x,y=tarzan_img.get_size()
tarzan_img = pygame.transform.scale(tarzan_img, (int(x*2), int(y*2)))

# --- YOLO setup ---#
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

class RunnerPoint:
    def __init__(self, x, y):
        self.x = float(x)
        self.y = float(y)
        self.vx = 0.0
        self.vy = 0.0
        self.facing_right = False  # Looks left by default

    def update(self, repulse_points, repulse_radius=300, repulse_strength=8.0, damping=0.92, wander=1,
               min_bounce_speed=4.0, corner_escape_radius=80, corner_escape_speed=16.0):
        # Check for corner escape
        corners = [(0, 0), (WIDTH, 0), (0, HEIGHT), (WIDTH, HEIGHT)]
        for cx, cy in corners:
            dist = math.hypot(self.x - cx, self.y - cy)
            if dist < corner_escape_radius:
                dx = self.x - cx
                dy = self.y - cy
                if dx == 0 and dy == 0:
                    dx, dy = 1, 1
                norm = math.hypot(dx, dy)
                self.vx = (dx / norm) * corner_escape_speed
                self.vy = (dy / norm) * corner_escape_speed
                self.x += self.vx
                self.y += self.vy
                if abs(self.vx) > 0.5:
                    self.facing_right = self.vx > 0
                return

        chased = False
        for rx, ry in repulse_points:
            dist = math.hypot(self.x - rx, self.y - ry)
            if repulse_radius > dist > 1:
                force = repulse_strength * (repulse_radius - dist) / (dist + 1)
                self.vx += (self.x - rx) / dist * force
                self.vy += (self.y - ry) / dist * force
                chased = True
        if not chased:
            angle = random.uniform(0, 2 * math.pi)
            self.vx += math.cos(angle) * wander
            self.vy += math.sin(angle) * wander
        self.vx *= damping
        self.vy *= damping
        self.x += self.vx
        self.y += self.vy

        bounced = False
        if self.x <= 0:
            self.x = 0
            self.vx = abs(self.vx)
            bounced = True
        elif self.x >= WIDTH:
            self.x = WIDTH
            self.vx = -abs(self.vx)
            bounced = True
        if self.y <= 0:
            self.y = 0
            self.vy = abs(self.vy)
            bounced = True
        elif self.y >= HEIGHT:
            self.y = HEIGHT
            self.vy = -abs(self.vy)
            bounced = True
        if bounced:
            self.vx += random.uniform(-4, 4)
            self.vy += random.uniform(-4, 4)
            speed = math.hypot(self.vx, self.vy)
            if speed < min_bounce_speed:
                angle = random.uniform(0, 2 * math.pi)
                self.vx = math.cos(angle) * min_bounce_speed
                self.vy = math.sin(angle) * min_bounce_speed

        if abs(self.vx) > 0.5:
            self.facing_right = self.vx > 0

    def draw(self, surface, radius=12):
        img = tarzan_img
        if self.facing_right:
            img = pygame.transform.flip(tarzan_img, True, False)
        img_rect = img.get_rect(center=(int(self.x), int(self.y)))
        surface.blit(img, img_rect)

runner = RunnerPoint(WIDTH // 2, HEIGHT // 2)

while True:
    with lock:
        ankles = latest_ankles.copy()
    repulse_points = [
        (int(ankle[0] * WIDTH / cap.get(3)), int(ankle[1] * HEIGHT / cap.get(4)))
        for left_ankle, right_ankle in ankles
        for ankle in [left_ankle, right_ankle]
    ]
    mouse = pygame.mouse.get_pos()
    repulse_points.append(mouse)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()

    screen.fill((0, 0, 0))
    runner.update(repulse_points)
    runner.draw(screen)
    pygame.display.flip()
    clock.tick(60)