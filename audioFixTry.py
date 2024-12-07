import cv2
import numpy as np
import random
import time
from ultralytics import YOLO
import pygame
import os


class RedLightGreenLightGame:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')

        # Initialize pygame mixer with higher frequency for better speed changes
        pygame.mixer.init(44100, -16, 2, 2048)

        # Load the sound files
        self.rlgl_sound = pygame.mixer.Sound('sounds/RLGLsong.mp3')
        self.win_sound = pygame.mixer.Sound('sounds/squidWin.mp3')
        self.lose_sound = pygame.mixer.Sound('sounds/kill.mp3')

        self.sound_playing = False
        self.current_speed = 1.0
        self.state_change_pending = False

        self.score = 0
        self.light = 'red'
        self.game_active = False
        self.last_detection = None
        self.movement_threshold = 30

        self.cap = cv2.VideoCapture(0)

        self.using_camera = False
        self.game_started = False

        # Load frame images
        self.frames = {
            'start': cv2.imread(os.path.join('frames', '4.png')),
            'win': cv2.imread(os.path.join('frames', '3.png')),
            'lose': cv2.imread(os.path.join('frames', '2.png'))
        }

        if any(frame is None for frame in self.frames.values()):
            raise ValueError("Failed to load one or more frame images")

        # Colors in BGR
        self.color_map = {
            'red': (0, 0, 255),
            'green': (0, 255, 0)
        }

    def start_state_change(self):
        if not pygame.mixer.get_busy():
            self.rlgl_sound.play()

            # Start sound timer
            self.sound_playing = True
            self.sound_start_time = time.time()

    def play_effect(self, sound):
        pygame.mixer.stop()
        sound.play()

    def cleanup_sound(self):
        pygame.mixer.stop()
        self.sound_playing = False
        self.state_change_pending = False

    def detect_movement(self, frame):
        results = self.model(frame, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.cls == 0:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    current_pos = (int((x1 + x2) / 2), int((y1 + y2) / 2))

                    if self.last_detection:
                        distance = np.sqrt(
                            (current_pos[0] - self.last_detection[0]) ** 2 +
                            (current_pos[1] - self.last_detection[1]) ** 2
                        )
                        if distance > self.movement_threshold:
                            return True

                    self.last_detection = current_pos
        return False

    def reset_game(self):
        self.cleanup_sound()
        self.score = 0
        self.light = 'red'
        self.game_active = True
        self.last_detection = None
        self.using_camera = True
        self.game_started = True

    def handle_game_over(self, frame):
        self.cleanup_sound()
        self.play_effect(self.lose_sound)
        self.using_camera = False
        self.game_active = False
        return self.frames['lose']

    def handle_win(self, frame):
        self.cleanup_sound()
        self.play_effect(self.win_sound)
        self.using_camera = False
        self.game_active = False
        return self.frames['win']

    def run_game(self):
        last_light_change = time.time()
        light_duration = random.uniform(2, 5)
        cv2.namedWindow('Red Light Green Light')

        while True:
            if not self.game_started:
                display_frame = self.frames['start'].copy()
                cv2.putText(display_frame, 'Press Q to start', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            elif self.using_camera:
                ret, display_frame = self.cap.read()
                if not ret:
                    break

                display_frame = cv2.flip(display_frame, 1)

                if self.game_active:
                    current_time = time.time()

                    # Start sound if not playing and enough time has passed
                    if not pygame.mixer.get_busy() and not self.sound_playing and \
                            current_time - last_light_change > light_duration:
                        self.start_state_change()

                    # Change state after sound finishes
                    if self.sound_playing and not pygame.mixer.get_busy():
                        self.sound_playing = False
                        self.light = 'green' if self.light == 'red' else 'red'
                        self.last_detection = None
                        last_light_change = current_time
                        light_duration = random.uniform(2, 5)

                    if self.detect_movement(display_frame):
                        if self.light == 'red':
                            display_frame = self.handle_game_over(display_frame)
                        else:
                            self.score += 1
                            if self.score >= 100:
                                display_frame = self.handle_win(display_frame)

                    cv2.putText(display_frame, f'Light: {self.light}', (50, 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, self.color_map[self.light], 2)
                    cv2.putText(display_frame, f'Score: {self.score}', (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    if pygame.mixer.get_busy():
                        # cv2.putText(display_frame, f'Speed: {self.current_speed:.1f}x',
                        #             (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        pass
            else:
                display_frame = self.frames['win'].copy() if self.score >= 100 else self.frames['lose'].copy()
                cv2.putText(display_frame, 'Press Q to try again', (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.imshow('Red Light Green Light', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                if not self.game_started:
                    self.game_started = True
                    self.using_camera = True
                    self.game_active = True
                elif not self.using_camera:
                    self.reset_game()
                else:
                    break

        self.cleanup_sound()
        pygame.mixer.quit()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    game = RedLightGreenLightGame()
    game.run_game()
