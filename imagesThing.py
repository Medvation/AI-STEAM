import cv2
import numpy as np
import random
import time
from ultralytics import YOLO
import soundfile as sf
import sounddevice as sd
import threading
import os


class RedLightGreenLightGame:
    def __init__(self):
        self.model = YOLO('yolov8n.pt')

        self.rlgl_sound, self.sample_rate = sf.read('sounds/RLGLsong.mp3')
        self.win_sound, _ = sf.read('sounds/squidWin.mp3')
        self.lose_sound, _ = sf.read('sounds/kill.mp3')

        self.sound_playing = False
        self.sound_start_time = 0
        self.current_speed = 1.0
        self.audio_thread = None
        self.state_change_pending = False
        self.stop_sound = threading.Event()

        self.score = 0
        self.light = 'red'
        self.game_active = False
        self.last_detection = None
        self.movement_threshold = 30

        self.cap = cv2.VideoCapture(0)

        # Game state
        self.using_camera = False
        self.game_started = False
        self.current_frame = '4.png'

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

    def play_sound(self, sound_data, speed=1.0):
        try:
            resampled_rate = int(self.sample_rate * speed)
            sd.play(sound_data, resampled_rate)
            while sd.get_stream().active and not self.stop_sound.is_set():
                sd.wait()
        except Exception as e:
            print(f"Sound playback error: {e}")
        finally:
            self.sound_playing = False
            if not self.stop_sound.is_set():
                self.state_change_pending = True
            self.stop_sound.clear()

    def start_state_change(self):
        if not self.sound_playing:
            self.current_speed = random.uniform(1, 2)
            self.sound_playing = True
            self.sound_start_time = time.time()
            self.stop_sound.clear()

            self.audio_thread = threading.Thread(
                target=self.play_sound,
                args=(self.rlgl_sound, self.current_speed)
            )
            self.audio_thread.daemon = True
            self.audio_thread.start()

    def play_effect(self, sound_data):
        if self.sound_playing:
            self.stop_sound.set()
            sd.stop()
            if self.audio_thread:
                self.audio_thread.join(timeout=1)
        sd.play(sound_data, self.sample_rate)
        sd.wait()

    def cleanup_sound(self):
        if self.sound_playing:
            self.stop_sound.set()
            sd.stop()
            if self.audio_thread:
                self.audio_thread.join(timeout=1)
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

                    if not self.sound_playing and not self.state_change_pending and current_time - last_light_change > light_duration:
                        self.start_state_change()

                    if self.state_change_pending:
                        self.light = 'green' if self.light == 'red' else 'red'
                        self.last_detection = None
                        last_light_change = current_time
                        light_duration = random.uniform(2, 5)
                        self.state_change_pending = False

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
                    self.cleanup_sound()
                    break

        self.cleanup_sound()
        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    game = RedLightGreenLightGame()
    game.run_game()
