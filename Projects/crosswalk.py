import pygame

pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
WIDTH, HEIGHT = screen.get_size()

def draw_crosswalk():
    screen.fill((0, 0, 0))
    margin = 0
    r_l_margin =350
    num_stripes = 10
    gap_between = 45  # space between stripes
    rect_width = WIDTH - 2 * margin
    rect_height = HEIGHT - 2 * margin
    total_gap = (num_stripes - 1) * gap_between
    stripe_height = (rect_height - total_gap)  // num_stripes

    y = margin

    for i in range(num_stripes):
        if i % 2 == 0:
            color = (255, 255, 255) # white for even stripes
        else:
            color = (255, 215, 0) #yellow for odd stripes

        #rect structure
                        #top left corner
                        #top right corner
                        #bottom left corner
                        #bottom right corner
        pygame.draw.rect(
            screen,
            color,
            (margin+r_l_margin, y, rect_width-(r_l_margin*2), stripe_height)
        )
        y += stripe_height + gap_between

    pygame.display.flip()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    draw_crosswalk()