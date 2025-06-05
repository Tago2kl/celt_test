import pygame
import math
import threading
import torch
import cv2
import numpy as np
import random
from ultralytics import YOLO
from PIL import Image

def load_gif_frames(path, size):
    frames = []
    try:
        pil_img = Image.open(path)
        for frame in range(0, getattr(pil_img, "n_frames", 1)):
            pil_img.seek(frame)
            frame_img = pil_img.convert("RGBA").resize(size, Image.LANCZOS)
            mode = frame_img.mode
            data = frame_img.tobytes()
            py_img = pygame.image.fromstring(data, size, mode)
            frames.append(py_img)
    except Exception as e:
        print(f"Error loading GIF frames from {path}: {e}")
    if not frames:
        frames.append(pygame.Surface(size, pygame.SRCALPHA))
    return frames

def hue_shift_surface(surface, hue_shift):
    arr = pygame.surfarray.pixels3d(surface).copy()
    arr = arr.astype(np.uint8)
    # Convert to HSV
    hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
    hsv[..., 0] = (hsv[..., 0].astype(int) + hue_shift) % 180
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    shifted = pygame.surfarray.make_surface(rgb)
    # Preserve alpha
    if surface.get_masks()[3] != 0:
        alpha = pygame.surfarray.pixels_alpha(surface)
        shifted = shifted.convert_alpha()
        pygame.surfarray.pixels_alpha(shifted)[:, :] = alpha
    return shifted

class Balloon:
    def __init__(self, idle_frames, pop_frames, screen_rect):
        self.base_idle_frames = idle_frames
        self.base_pop_frames = pop_frames
        self.screen_rect = screen_rect
        self.size = idle_frames[0].get_size()
        self.respawn()

    def _make_scaled_frames(self):
        scale = 4
        size = (self.size[0] * scale, self.size[1] * scale)
        self.idle_frames = [
            hue_shift_surface(pygame.transform.scale(f, size), self.hue_shift)
            for f in self.base_idle_frames
        ]
        self.pop_frames = [
            hue_shift_surface(pygame.transform.scale(f, size), self.hue_shift)
            for f in self.base_pop_frames
        ]
        self.current_size = size

    def respawn(self):
        self.hue_shift = random.randint(0, 179)
        self._make_scaled_frames()
        side = random.choice(['left', 'right'])
        if side == 'left':
            self.x = -self.current_size[0]
            self.vx = abs(random.choice([2, 3]))
        else:
            self.x = self.screen_rect.width
            self.vx = -abs(random.choice([2, 3]))
        self.y = random.randint(0, self.screen_rect.height - self.current_size[1])
        self.vy = random.uniform(-1, 1)
        self.state = "idle"
        self.frame_idx = 0
        self.pop_timer = 0

    def update(self):
        if self.state == "idle":
            self.x += self.vx
            self.y += self.vy
            if not self.screen_rect.colliderect(pygame.Rect(self.x, self.y, *self.current_size)):
                self.respawn()
            self.frame_idx = (self.frame_idx + 1) % len(self.idle_frames)
        elif self.state == "pop":
            self.pop_timer += 1
            if self.pop_timer >= len(self.pop_frames):
                self.respawn()
            else:
                self.frame_idx = self.pop_timer

    def draw(self, surface):
        if self.state == "idle":
            frame = self.idle_frames[self.frame_idx]
        else:
            idx = min(self.frame_idx, len(self.pop_frames) - 1)
            frame = self.pop_frames[idx]
        surface.blit(frame, (self.x, self.y))

    def rect(self):
        return pygame.Rect(self.x, self.y, *self.current_size)

    def pop(self):
        if self.state != "pop":
            self.state = "pop"
            self.frame_idx = 0
            self.pop_timer = 0

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
WIDTH, HEIGHT = screen.get_size()
clock = pygame.time.Clock()
screen_rect = pygame.Rect(0, 0, WIDTH, HEIGHT)

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

# --- Balloon setup ---
balloon_size = (100, 100)
idle_frames = load_gif_frames("Assets/idle_bloon.gif", balloon_size)
pop_frames = load_gif_frames("Assets/pop_bloon.gif", balloon_size)
balloons = [Balloon(idle_frames, pop_frames, screen_rect) for _ in range(5)]

# --- Main Loop ---
running = True
while running:
    screen.fill((25,25, 25))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get ankle points
    with lock:
        ankles = latest_ankles.copy()
    repulse_points = [
        (int(ankle[0] * WIDTH / cap.get(3)), int(ankle[1] * HEIGHT / cap.get(4)))
        for left_ankle, right_ankle in ankles
        for ankle in [left_ankle, right_ankle]
    ]
    mouse = pygame.mouse.get_pos()

    # Update and draw balloons
    for balloon in balloons:
        balloon.update()
        balloon.draw(screen)
        # Pop if mouse or any ankle is above
        if balloon.state == "idle":
            brect = balloon.rect()
            if brect.collidepoint(mouse):
                balloon.pop()
            else:
                for pt in repulse_points:
                    if brect.collidepoint(pt):
                        balloon.pop()
                        break

    pygame.display.flip()
    clock.tick(30)

cap.release()
pygame.quit()