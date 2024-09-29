import cv2
import mediapipe as mp
import random
import time
import warnings

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

webcam = cv2.VideoCapture(0)
width = 1920
height = 1200
target_size = 50
hit_range = 100

webcam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

levels = {
    "easy": {"lives": 10, "timeout": 4, "win_score": 25},
    "medium": {"lives": 5, "timeout": 3, "win_score": 50},
    "hard": {"lives": 3, "timeout": 2, "win_score": 100}
}

score = 0
lives = 0
target_visible = False
start_time = None
difficulty = None
target_pos = None

def get_random_position():
    x = random.randint(width // 2 - 200, width // 2 + 200 - target_size)
    y = random.randint(height // 2 - 200, height // 2 + 200 - target_size)
    return (x, y)

def draw_state(frame):
    cv2.putText(frame, f"Score: {score}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(frame, f"Lives: {lives}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

def choose_difficulty():
    global difficulty, lives, start_time, target_visible, target_pos
    target_visible = False
    options = {
        "easy": (width // 3, height // 2 - 50),
        "medium": (width // 2, height // 2 - 50),
        "hard": (2 * width // 3, height // 2 - 50)
    }
    
    while difficulty is None:
        ret, frame = webcam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(frame_rgb)

        for level, (x, y) in options.items():
            cv2.putText(frame, level.capitalize(), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                fingertip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                fingertip_x = int(fingertip.x * width)
                fingertip_y = int(fingertip.y * height)

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                cv2.circle(frame, (fingertip_x, fingertip_y), 10, (0, 0, 255), -1)

                for level, (x, y) in options.items():
                    if x - 50 < fingertip_x < x + 50 and y - 30 < fingertip_y < y + 30:
                        difficulty = level
                        lives = levels[difficulty]["lives"]
                        start_time = time.time()
                        target_pos = get_random_position()
                        target_visible = True
                        break

        cv2.imshow("Choose Difficulty", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

def game_loop():
    global score, lives, target_visible, start_time, target_pos

    while lives > 0:
        ret, frame = webcam.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(frame_rgb)

        if target_visible:
            if time.time() - start_time < levels[difficulty]["timeout"]:
                cv2.rectangle(frame, target_pos, (target_pos[0] + target_size, target_pos[1] + target_size), (0, 255, 0), -1)
            else:
                target_visible = False
                lives -= 1  # Lose a life if time runs out
                target_pos = get_random_position()
                start_time = time.time()  # Reset start time
                target_visible = True  # Show the new target immediately

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                fingertip = hand.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                fingertip_x = int(fingertip.x * width)
                fingertip_y = int(fingertip.y * height)

                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                cv2.circle(frame, (fingertip_x, fingertip_y), 10, (0, 0, 255), -1)

                if (target_pos[0] - hit_range < fingertip_x < target_pos[0] + target_size + hit_range) and \
                   (target_pos[1] - hit_range < fingertip_y < target_pos[1] + target_size + hit_range):
                    target_pos = get_random_position()
                    target_visible = True  # Show the target again
                    score += 1
                    start_time = time.time()  # Reset start time on hit

        draw_state(frame)

        if score >= levels[difficulty]["win_score"]:
            cv2.putText(frame, "You Win!", (width // 2 - 100, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
            cv2.imshow("Finger Tracking Game", frame)
            cv2.waitKey(3000)
            break

        cv2.imshow("Finger Tracking Game", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    if lives <= 0:
        cv2.putText(frame, "Game Over!", (width // 2 - 100, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.imshow("Finger Tracking Game", frame)
        cv2.waitKey(3000)

choose_difficulty()
game_loop()

webcam.release()
cv2.destroyAllWindows()
warnings.filterwarnings('ignore')
