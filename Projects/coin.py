
import cv2
import pygame
from ultralytics import YOLO
import threading

def draw_coin_bg(screen, frame_idx):
    coin_img = coin_frames[frame_idx]
    bg_img = pygame.transform.smoothscale(coin_img, (1280, 720))
    screen.blit(bg_img, (0, 0))

def wave(people_count, spin_speed, min_speed, max_speed, idle_speed, ramp_timer, speedup_frames):
    slowdown_rate = 0.02
    if people_count == 0:
        return idle_speed
    if ramp_timer < speedup_frames:
        speedup_rate = (max_speed - spin_speed) / (speedup_frames - ramp_timer) if ramp_timer < speedup_frames else 0
        spin_speed += speedup_rate
        if spin_speed > max_speed:
            spin_speed = max_speed
    else:
        spin_speed -= slowdown_rate
        if spin_speed < idle_speed:
            spin_speed = idle_speed
    return spin_speed

# In your main loop, replace rampcount logic with:

def yolo_thread_func(cap, model):
    global latest_keypoints, people_count
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            results = model(frame, conf=0.3, stream=True, show=True)
            new_keypoints = []
            count = 0
            for result in results:
                for person_kpts in result.keypoints.xy:
                    if person_kpts.shape[0] >= 17:
                        left_ankle = person_kpts[15]
                        right_ankle = person_kpts[16]
                        new_keypoints.append((left_ankle, right_ankle))
                        count += 1
            with lock:
                latest_keypoints = new_keypoints
                people_count = count
    except Exception as e:
        print(f"YOLO thread error: {e}")

# --- Load coin video frames ---#
coin_cap = cv2.VideoCapture('../../../Documents/codefolder/pycharmtest/opencv/assets/spincoincole.mp4')
coin_frames = []

while True:
    ret, frame = coin_cap.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    frame = cv2.resize(frame, (1280, 720))
    surf = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], 'RGBA')
    coin_frames.append(surf)
coin_cap.release()

if not coin_frames:
    print("Error: No frames loaded from spincoincole.mp4")
    exit(1)

coin_frame_idx = 0
coin_frame_count = len(coin_frames)

# --- Initialize Pygame and other variables ---
pygame.init()
screen = pygame.display.set_mode((1280, 720))
clock = pygame.time.Clock()

# --- YOLO and threading setup ---
try:
    model = YOLO("../../../Documents/codefolder/pycharmtest/opencv/models/yolo11n-pose.pt")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit(1)

cap = cv2.VideoCapture("../../../Documents/codefolder/pycharmtest/opencv/assets/ppl.webm")
if not cap.isOpened():
    print("Error: Could not open video file ppl.webm")
    exit(1)

latest_keypoints = []
people_count = 0
lock = threading.Lock()

threading.Thread(target=yolo_thread_func, args=(cap, model), daemon=True).start()

spin_speed = 1.0
prev_people_count = 0
min_speed = 0.5
max_speed = 12.0
half_speed = (max_speed + min_speed) / 2
idle_speed = 1.0  # Set idle speed to 1
rampcount = 0
ramp_timer = 0
speedup_frames = 120.0
coin_frame_pos = 0.0

while True:
    if people_count == 0:
        ramp_timer = 0
    elif people_count > prev_people_count:
        ramp_timer = 0  # Reset ramp for new arrivals
    elif ramp_timer < speedup_frames:
        ramp_timer += 1
    spin_speed = wave(people_count, spin_speed, min_speed, max_speed, idle_speed, ramp_timer, speedup_frames)
    print(int(spin_speed))
    coin_frame_pos = (coin_frame_pos + spin_speed) % coin_frame_count
    coin_frame_idx = int(coin_frame_pos)

    prev_people_count = people_count
    draw_coin_bg(screen, coin_frame_idx)

    with lock:
        for left_ankle, right_ankle in latest_keypoints:
            for ankle in [left_ankle, right_ankle]:
                x = int(ankle[0] * 1280 / cap.get(3))
                y = int(ankle[1] * 720 / cap.get(4))
                pygame.draw.circle(screen, (255, 255, 255), (x, y), 10)
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            exit()
    clock.tick(60)