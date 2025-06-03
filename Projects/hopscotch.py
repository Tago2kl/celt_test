
import pygame
import threading
import cv2
import time
import random
from ultralytics import YOLO

# --- Initialization ---
pygame.init()
font = pygame.font.Font(None, 100)
font_color = (0, 200, 0)
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
WIDTH, HEIGHT = screen.get_size()
clock = pygame.time.Clock()

# --- YOLO and Video Setup ---
model = YOLO("../../../Documents/codefolder/pycharmtest/opencv/models/yolo11n-pose.pt")
cap = cv2.VideoCapture("../../../Documents/codefolder/pycharmtest/opencv/assets/pplwalk.mp4")
latest_ankles = []
lock = threading.Lock()

# Load Tarzan image
tarzan_img = pygame.image.load('../../../Documents/codefolder/pycharmtest/opencv/assets/img.png')
tarzan_img = pygame.transform.scale(tarzan_img, (WIDTH, HEIGHT))

def generate_hopscotch_pattern(num_spaces=6):
    """generate a hopscotch pattern with at least 2 doubles &  never 3 doubles in a row."""
    pattern = []
    for jump in range(num_spaces):
        if len(pattern) >= 2 and pattern[-1] == 2 and pattern[-2] == 2:
            p = 1
        else:
            p = random.choices([1, 2], weights=[2, 1])[0]
        pattern.append(p)

    while pattern.count(2) < 2:
        candidates = [
            i for i in range(num_spaces)
            if pattern[i] == 1 and
               (i < 2 or not (pattern[i - 1] == 2 and pattern[i - 2] == 2)) and
               (i > num_spaces - 3 or not (pattern[i + 1] == 2 and pattern[i + 2] == 2))
        ]
        if not candidates:
            break
        i = random.choice(candidates)
        pattern[i] = 2
    return pattern

def draw_hopscotch(surface, pattern, green_boxes):
    """draw the hopscotch board"""
    num_spaces = len(pattern)
    min_gap = 18
    space_h = (HEIGHT - (num_spaces - 1) * min_gap) // num_spaces
    space_w = WIDTH // 5
    x_center = WIDTH // 2
    y = 0
    box_num = sum(pattern)
    box_rects = []

    for p in pattern:
        if p == 1:
            rect = pygame.Rect(x_center - space_w // 2, y, space_w, space_h)
            color = (0, 255, 0) if green_boxes[len(box_rects)] else (255, 255, 255)
            pygame.draw.rect(surface, color, rect, border_radius=10)
            draw_number_with_border(surface, rect.center, box_num)
            box_rects.append(rect)
            box_num -= 1
        else:
            rect1 = pygame.Rect(x_center - space_w - 12, y, space_w, space_h)
            rect2 = pygame.Rect(x_center + 12, y, space_w, space_h)
            color1 = (0, 255, 0) if green_boxes[len(box_rects)] else (255, 255, 255)
            color2 = (0, 255, 0) if green_boxes[len(box_rects)+1] else (255, 255, 255)
            pygame.draw.rect(surface, color1, rect1, border_radius=10)
            pygame.draw.rect(surface, color2, rect2, border_radius=10)
            draw_number_with_border(surface, rect1.center, box_num - 1)
            draw_number_with_border(surface, rect2.center, box_num)
            box_rects.append(rect1)
            box_rects.append(rect2)
            box_num -= 2
        y += space_h
        if box_num > 0:
            y += min_gap
    return box_rects

def draw_number_with_border(surface, center, number):
    """Draw a number with a thick black border for visibility."""
    for dx in [-2, -1, 0, 1, 2]:
        for dy in [-2, -1, 0, 1, 2]:
            if dx != 0 or dy != 0:
                border_text = font.render(str(number), True, (0, 0, 0))
                text_rect = border_text.get_rect(center=center)
                text_rect.move_ip(dx, dy)
                surface.blit(border_text, text_rect)
    text = font.render(str(number), True, font_color)
    text_rect = text.get_rect(center=center)
    surface.blit(text, text_rect)

def get_ankle_points():
    """get ankle points"""
    with lock:
        ankles = latest_ankles.copy()
    points = [
        (int(ankle[0] * WIDTH / cap.get(3)), int(ankle[1] * HEIGHT / cap.get(4)))
        for left_ankle, right_ankle in ankles
        for ankle in [left_ankle, right_ankle]
    ]
    return points

def update_green_boxes(box_rects, ankle_points, green_boxes):
    """Update which boxes are green based on ankle positions."""
    for i, rect in enumerate(box_rects):
        for ax, ay in ankle_points:
            if rect.collidepoint(ax, ay):
                green_boxes[i] = True

running = True
finish = False
pattern = generate_hopscotch_pattern()
green_boxes = [False] * sum([1 if p == 1 else 2 for p in pattern])

while running:
    screen.fill((0, 0, 0))
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                finish = False
                pattern = generate_hopscotch_pattern()
                green_boxes = [False] * sum([1 if p == 1 else 2 for p in pattern])


    if not finish:
        ankle_points = get_ankle_points()
        mouse = pygame.mouse.get_pos()
        box_rects = draw_hopscotch(screen, pattern, green_boxes)
        if len(green_boxes) != len(box_rects):
            green_boxes = [False] * len(box_rects)
        update_green_boxes(box_rects, ankle_points, green_boxes)
        update_green_boxes(box_rects, [mouse], green_boxes)
        if all(green_boxes) and len(green_boxes) > 0:
            finish = True  # Only set finish here!
        pygame.display.flip()
        clock.tick(60)
    else:
        # Reset after showing Tarzan
        finish = False
        pattern = generate_hopscotch_pattern()
        green_boxes = [False] * sum([1 if p == 1 else 2 for p in pattern])

cap.release()
pygame.quit()


