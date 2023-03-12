import pygame_gui
import pygame.camera
from ransac import Ransac_Demo
import os


def create_directory(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


create_directory("src/output")
create_directory("src/input")

pygame.init()

pygame.display.set_caption('Quick Start')
window_surface = pygame.display.set_mode((1024, 576))

imp = pygame.image.load("img.png").convert()

# Using blit to copy content from one surface to other
window_surface.blit(imp, (0, 0))

manager = pygame_gui.UIManager((1024, 576))

font = pygame.font.Font('freesansbold.ttf', 32)

# create a text surface object,
# on which text is drawn on it.
text = font.render('RANSAC algorithm for image alignment', True, (255, 0, 255))

# create a rectangular object for the
# text surface object
textRect = text.get_rect()

textRect.center = (1024 // 2, 25)

Turn_on_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((775, 150), (200, 50)),
                                              text='Turn on camera',
                                              manager=manager)

Turn_off_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((775, 200), (200, 50)),
                                               text='Turn off camera',
                                               manager=manager)

Capture_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((775, 250), (200, 50)),
                                              text='Take a photo',
                                              manager=manager)

Show_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((775, 300), (200, 50)),
                                           text='Show a result',
                                           manager=manager)

Hide_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((775, 350), (200, 50)),
                                           text='Hide a result',
                                           manager=manager)

clock = pygame.time.Clock()
is_running = True
cam = None
result = None
img = None
count = 0

while is_running:
    window_surface.blit(text, textRect)

    if cam:
        img = cam.get_image()
        window_surface.blit(img, (50, 75))

    if result:
        window_surface.blit(result, (50, 75))

    pygame.display.update()

    window_surface.blit(imp, (0, 0))
    manager.draw_ui(window_surface)

    time_delta = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False
            exit()

        if event.type == pygame_gui.UI_BUTTON_PRESSED:
            if event.ui_element == Turn_on_button:
                pygame.camera.init()
                cur_cam = pygame.camera.list_cameras()
                cam = pygame.camera.Camera(cur_cam[0], (640, 480))
                cam.start()

            if event.ui_element == Turn_off_button:
                cam.stop()
                cam = None

            if event.ui_element == Capture_button and cam:
                pygame.image.save(img, f'src/input/filename_{count}.jpg')
                count += 1
            if event.ui_element == Show_button:
                Ransac_Demo()
                result = pygame.image.load("ransac_img.jpg").convert()

            if event.ui_element == Hide_button:
                result = None

        manager.process_events(event)

    manager.update(time_delta)
