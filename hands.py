import time
import math
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Debugging 
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
prev_frame_time = 0
new_frame_time = 0
debug_msg = \
  "FPS: {:.0f} " + \
  "Distance to camera: {:.2f}"
status_msg = "No hand detected"

# Gesture detection
distance_to_camera = 0

# Capture from webcam input
cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.95,
    min_tracking_confidence=0.85) as hands:
  while cap.isOpened():
    success, image = cap.read()
    height, width, channels = image.shape

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        # Compute the distance between each fingers to determine if hand is open or closed
        wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
        pinky = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
        distance_to_camera = math.sqrt( (wrist.x - pinky.x) ** 2 + (wrist.y - pinky.y) ** 2 )

        # Draw handlandmarks
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # Calculate FPS here using time
    new_frame_time = time.time()
    fps = 1 / (new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time

    # Final processing of frame
    image_flipped = cv2.flip(image, 1)
    image_flipped = cv2.putText(
      image_flipped, debug_msg.format(fps, distance_to_camera),
      (0, height - 12), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('Hands', image_flipped)
    if cv2.waitKey(5) & 0xFF == 27:
      break

cap.release()
