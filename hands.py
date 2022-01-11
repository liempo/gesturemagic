import time
import itertools

import cv2 as cv
import mediapipe as mp
import pyautogui


from model.gesture_classifier import \
  Gesture, GestureClassifier

def main(): 
  # Initialize mediapipe 
  mp_drawing = mp.solutions.drawing_utils
  mp_drawing_styles = mp.solutions.drawing_styles
  mp_hands = mp.solutions.hands
  hands = mp_hands.Hands(
      model_complexity=0,
      max_num_hands=1,
      min_detection_confidence=0.85,
      min_tracking_confidence=0.75)

  # Create debugging utilities
  prev_frame_time = 0
  new_frame_time = 0
  debug_msg = "FPS: {:.0f}"
  status_msg = "Gesture: {}"
  font = cv.FONT_HERSHEY_COMPLEX_SMALL

  # Create gesture classifier
  classify = GestureClassifier()
  gesture = None

  # OpenCV start video capture
  cap = cv.VideoCapture(0)
  cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

  last_gesture_recognize = None

  while cap.isOpened():
    success, image = cap.read()

    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue
    
    image_height, image_width, _ = image.shape

    # Disable writeable to improve performance
    image.flags.writeable = False
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:

        preprocessed_landmarks = preprocess_landmarks(
          hand_landmarks, image_width, image_height)
        gesture = classify(preprocessed_landmarks)
        
        #Shortcut hotkeys
        if gesture != last_gesture_recognize:
          last_gesture_recognize = gesture
          if gesture == Gesture.OPEN:
            pyautogui.hotkey("ctrlleft", "c")
          elif gesture == Gesture.CLOSED:
            pyautogui.hotkey("ctrlleft", "v")
          elif gesture == Gesture.OKAY:
            pyautogui.hotkey("ctrlleft", "x")   

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
    image_flipped = cv.flip(image, 1)
    image_flipped = cv.putText(
      image_flipped, debug_msg.format(fps),
      (0, image_height - 12), font, 0.5,
      (255, 255, 255), 1, cv.LINE_AA)
    image_flipped = cv.putText(
      image_flipped, status_msg.format(gesture),
       (0, 24), font, 1.5,
      (255, 255, 255), 1, cv.LINE_AA)
    
    # Flip the image horizontally for a selfie-view display.
    cv.imshow('Hands', image_flipped)
    if cv.waitKey(5) & 0xFF == 27:
      break

  cap.release()

def preprocess_landmarks(landmarks, image_width, image_height):
  # Transofrm landmark into absolute points 
  absolute_points = []
  for landmark in landmarks.landmark:
    x = min(int(landmark.x * image_width), image_width - 1)
    y = min(int(landmark.y * image_height), image_height - 1)
    absolute_points.append([x, y])
  
  if len(absolute_points) == 0:
    return absolute_points

  # Transform absolute points into relative points
  relative_points = []
  base_x, base_y = 0, 0
  for index, point in enumerate(absolute_points):
    if index == 0:
      base_x, base_y = point[0], point[1]
    x = point[0] - base_x
    y = point[1] - base_y
    relative_points.append([x, y])
  
  # Convert to a one-dimensional list
  points = list(itertools.chain.from_iterable(relative_points))

  # Normalize the values
  max_value = max(list(map(abs, points)))
  def _normalize(n):
    return n / max_value
  points = list(map(_normalize, points))

  return points

if __name__ == "__main__":
  main()