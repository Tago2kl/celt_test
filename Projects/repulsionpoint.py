import pygame
import random
import math
import threading
import torch
import cv2
import numpy as np
from ultralytics import YOLO

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
WIDTH, HEIGHT = screen.get_size()
NUM_PARTICLES = 100
MAX_DISTANCE = 250
PARTICLE_SPEED = 15
REPULSE_RADIUS = 250

clock = pygame.time.Clock()

# --- YOLO setup ---#
model = YOLO("Detection_Models/yolo11n-pose.pt")
cap = cv2.VideoCapture("Assets/pplwalk.mp4")
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

# chech for gpu availability
if torch.cuda.is_available():
    print(torch.cuda.is_available())
    model.to('cuda')

threading.Thread(target=yolo_thread_func, daemon=True).start()

class Particle:
    def __init__(self):
        self.x = random.uniform(0, WIDTH)
        self.y = random.uniform(0, HEIGHT)
        angle = random.uniform(0, 2 * math.pi)
        self.radius = random.randint(2, 4)
        speed = PARTICLE_SPEED * (1 / self.radius)
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed

    def move(self):
        self.x += self.vx
        self.y += self.vy
        if self.x <= 0 or self.x >= WIDTH:
            self.vx *= -1
        if self.y <= 0 or self.y >= HEIGHT:
            self.vy *= -1

        # limit velocity
        self.vx = max(min(self.vx, 2), -2)
        self.vy = max(min(self.vy, 2), -2)

    def draw(self, surface):
        pygame.draw.circle(surface, (255, 255, 255), (int(self.x), int(self.y)), self.radius)

    def repulse_from(self, points):
        for mx, my in points:
            dx = self.x - mx
            dy = self.y - my
            dist = math.hypot(dx, dy)
            if dist < REPULSE_RADIUS and dist != 0:
                force = (REPULSE_RADIUS - dist) / REPULSE_RADIUS
                self.vx += (dx / dist) * force
                self.vy += (dy / dist) * force

particles = []
for _ in range(NUM_PARTICLES):
    particle = Particle()  # create a new particle object
    particles.append(particle)

running = True
line_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

while running:
    screen.fill((0, 0, 0,0))
    line_surface.fill((0, 0, 0, 0))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    with lock:
        ankles = latest_ankles.copy()
    repulse_points = [
        (int(ankle[0] * WIDTH / cap.get(3)), int(ankle[1] * HEIGHT / cap.get(4)))
        for left_ankle, right_ankle in ankles
        for ankle in [left_ankle, right_ankle]
    ]

    # draw ankle points once
    for mx, my in repulse_points:
        pygame.draw.circle(screen, (255, 255, 255), (mx, my), 12, 10)
        continue

    for p in particles:
        p.repulse_from(repulse_points)
        p.move()
        p.draw(screen)

    # Draw lines between close particles
    for i, p1 in enumerate(particles):
        for p2 in particles[i+1:]:
            dx = p1.x - p2.x
            dy = p1.y - p2.y
            dist = math.hypot(dx, dy)
            if dist < MAX_DISTANCE:
                alpha = int(255 * (1 - dist / MAX_DISTANCE))
                color = (255, 255, 255, alpha)
                pygame.draw.line(line_surface, color, (p1.x, p1.y), (p2.x, p2.y), 1)

    screen.blit(line_surface, (0, 0))
    pygame.display.flip()
    clock.tick(60)

cap.release()
pygame.quit()