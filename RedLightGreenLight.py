import time
import threading
import random
from playsound import playsound
import cv2
import os

# Base directory for assets
#BASE_DIR = '/Users/sakhaaalsaedi/Desktop/HS_CSEdWEEK/red-light-green-light/'

# Paths for frames and sounds
# FRAMES_DIR = os.path.join(BASE_DIR, 'frames')
# SOUNDS_DIR = os.path.join(BASE_DIR, 'sounds')
FRAMES_DIR =  'frames'
SOUNDS_DIR =  'sounds'

# Load images and sounds
images = [cv2.imread(os.path.join(FRAMES_DIR, img)) for img in sorted(os.listdir(FRAMES_DIR))]
green, red, kill, winner, intro = images[:5]
win_sound = os.path.join(SOUNDS_DIR, 'squidWin.mp3')
kill_sound = os.path.join(SOUNDS_DIR, 'kill.mp3')
green_sound = os.path.join(SOUNDS_DIR, 'RLGLsong.mp3')

# Variables
GAME_DURATION = 10
GAME_DURATION, RED_STATE_DURATION = 10, 1
state, score, game_running = "green", 0, True
font = cv2.FONT_HERSHEY_SIMPLEX

# Green light music thread
def green_music():
    global state
    while game_running:
        if state == "green":
            playsound(green_sound, block=False)  # Play green light music
            time.sleep(random.randint(5, 10))  # Randomize green state duration
            state = "red"


cv2.imshow('Squid Game', cv2.resize(intro, (0, 0), fx=0.5, fy=0.5))
cv2.waitKey(3000)

# Start game
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Camera not accessible")
threading.Thread(target=green_music).start()

end_time = time.time() + GAME_DURATION
while time.time() < end_time:
    ret, frame = cap.read()
    if not ret:
        break

    remaining_time = int(end_time - time.time())
    frame = cv2.resize(green if state == "green" else red, (0, 0), fx=0.5, fy=0.5)

    if state == "red":  # Red state: detect 'W' key press
        start_time = time.time()
        while time.time() - start_time < RED_STATE_DURATION:
            cv2.imshow('Squid Game', frame)
            if cv2.waitKey(1) & 0xFF == ord('w'):  # Kill if 'W' pressed during red
                cv2.imshow('Squid Game', cv2.resize(kill, (0, 0), fx=0.5, fy=0.5))
                cv2.waitKey(1000)  # Show kill frame for 1 second
                playsound(kill_sound)
                state = "kill"
                break
        if state != "kill":
            state = "green"

    elif state == "green":  # Green state: count 'W' key presses
        if cv2.waitKey(10) & 0xFF == ord('w'):
            score += 1

    # Display timer and score
    cv2.putText(frame, f"Time: {remaining_time}s", (20, 30), font, 1, (0, 255, 255), 2)
    cv2.putText(frame, f"Score: {score}", (20, 70), font, 1, (0, 255, 255), 2)
    cv2.imshow('Squid Game', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):  # Quit on 'Q' key press
        state = "kill"
        break

# Cleanup
game_running = False
cap.release()
cv2.destroyAllWindows()

# Game result
if state == "kill":  # Show kill frame before exiting
    cv2.imshow('Squid Game', cv2.resize(kill, (0, 0), fx=0.5, fy=0.5))
    cv2.waitKey(2000)  # Display kill frame for 2 seconds
    playsound(kill_sound)
else:  # Display win or lose screen based on score
    result = winner if score > 15 else kill
    cv2.imshow('Squid Game', cv2.resize(result, (0, 0), fx=0.5, fy=0.5))
    playsound(win_sound if score > 15 else kill_sound)
    cv2.waitKey(5000)

print(f"Game Over! Your Score: {score}")
