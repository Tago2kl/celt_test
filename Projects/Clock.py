import pygame
import sys
import math
import datetime
import threading
from ultralytics import YOLO
import cv2
import numpy as np

# --- YOLO and Video Setup ---
model = YOLO("Detection_Models/yolo11n-pose.pt")
cap = cv2.VideoCapture("Assets/pplwalk.mp4")
people_count = 0
lock = threading.Lock()
prev_ankles = []

def is_new_person(new_ankle, prev_ankles, threshold=50):
    new_ankle_np = np.array(new_ankle)
    for prev in prev_ankles:
        prev_np = np.array(prev)
        if np.linalg.norm(new_ankle_np - prev_np) < threshold:
            return False
    return True

def yolo_thread_func():
    global people_count, prev_ankles
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        results = model(frame)
        new_ankles = []
        for result in results:
            kpts = result.keypoints.xy if hasattr(result, "keypoints") and result.keypoints is not None else []
            if len(kpts) > 0:
                for person in kpts:
                    if person.shape[0] >= 17:
                        left_ankle = person[15].tolist()
                        if is_new_person(left_ankle, prev_ankles):
                            with lock:
                                people_count += 1
                        new_ankles.append(left_ankle)
        prev_ankles = new_ankles

threading.Thread(target=yolo_thread_func, daemon=True).start()

# --- Pygame Setup ---
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
WIDTH, HEIGHT = screen.get_size()
clock = pygame.time.Clock()

font_time = pygame.font.Font("font/Montserrat-Bold.ttf", 120)
font_date = pygame.font.Font("font/Montserrat-Medium.ttf", 60)

time_offset = 0

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT or \
           (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            cap.release()
            pygame.quit()
            sys.exit()

    screen.fill((0, 0, 0))

    now = datetime.datetime.now() + datetime.timedelta(hours=time_offset)
    hour_12 = now.strftime("%I").lstrip("0") or "12"
    minute = now.strftime("%M")
    second = now.strftime("%S")
    ampm = now.strftime("%p")
    time_str = f"{hour_12}:{minute}:{second} {ampm}"
    date_str = now.strftime("%A, %d %B %Y")

    # Render time and date
    time_surf = font_time.render(time_str, True, (255, 255, 255))
    time_rect = time_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 60))
    screen.blit(time_surf, time_rect)
    date_surf = font_date.render(date_str, True, (200, 200, 220))
    date_rect = date_surf.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 80))
    screen.blit(date_surf, date_rect)

    # --- Draw sun or moon in a full circle around the center ---
    center_x = WIDTH // 2
    center_y = (time_rect.centery + date_rect.centery) // 2
    radius = max(time_rect.width, date_rect.width) // 2 + 140

    # Angle: 0 at top, increases clockwise, 0-24h -> 0-360deg
    total_seconds = now.hour * 3600 + now.minute * 60 + now.second
    angle = (total_seconds / (24 * 3600)) * 2 * math.pi - math.pi / 2  # -90deg so 0h is at top

    sun_x = int(center_x + radius * math.cos(angle))
    sun_y = int(center_y + radius * math.sin(angle))

    center_x = WIDTH // 2
    center_y = (time_rect.centery + date_rect.centery) // 2
    radius = max(time_rect.width, date_rect.width) // 2 + 140

    hour = now.hour
    minute = now.minute
    second = now.second

    if 6 <= hour < 18:
        # Daytime: sun moves on upper semicircle (left to right)
        day_seconds = (hour - 6) * 3600 + minute * 60 + second
        angle = math.pi + (day_seconds / (12 * 3600)) * math.pi  # pi (left) to 2pi (right)
        # Draw sun
        sun_x = int(center_x + radius * math.cos(angle))
        sun_y = int(center_y + radius * math.sin(angle))
        pygame.draw.circle(screen, (255, 220, 80), (sun_x, sun_y), 60)
        for r, alpha in [(90, 40), (120, 20)]:
            glow = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
            pygame.draw.circle(glow, (255, 220, 80, alpha), (r, r), r)
            screen.blit(glow, (sun_x - r, sun_y - r), special_flags=pygame.BLEND_RGBA_ADD)
    else:
        # Nighttime: moon moves on lower semicircle (right to left)
        if hour < 6:
            night_seconds = (hour + 6) * 3600 + minute * 60 + second  # 0-6h -> 6-12h
        else:
            night_seconds = (hour - 18) * 3600 + minute * 60 + second  # 18-24h -> 0-6h
        angle = 0 + (night_seconds / (12 * 3600)) * math.pi  # 0 (right) to pi (left)
        # Draw moon
        moon_x = int(center_x + radius * math.cos(angle))
        moon_y = int(center_y + radius * math.sin(angle))
        pygame.draw.circle(screen, (200, 200, 255), (moon_x, moon_y), 60)
        crescent = pygame.Surface((120, 120), pygame.SRCALPHA)
        pygame.draw.circle(crescent, (0, 0, 0, 0), (60, 60), 60)
        pygame.draw.circle(crescent, (0, 0, 0, 255), (80, 60), 60)
        screen.blit(crescent, (moon_x - 60, moon_y - 60), special_flags=pygame.BLEND_RGBA_SUB)
    pygame.display.flip()
    clock.tick(60)