# -*- coding: utf-8 -*-
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
cap.set(cv2.CAP_PROP_FPS, 30)

mp_hands = mp.solutions.hands
hand = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

pTime = 0  # FPS için önceki zaman

# Parmak uçları: Thumb, Index, Middle, Ring, Pinky
tip_ids = [4, 8, 12, 16, 20]

while True:
    success, frame = cap.read()
    if not success:
        break
     
    frame = cv2.flip(frame,1)
    RGB_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hand.process(RGB_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            fingers = []

            # Baş parmak (sağ el için x koordinatına göre)
            if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0]-1].x:
                fingers.append(1)
            else:
                fingers.append(0)

            # Diğer parmaklar (y koordinatına göre)
            for id in range(1,5):
                if hand_landmarks.landmark[tip_ids[id]].y < hand_landmarks.landmark[tip_ids[id]-2].y:
                    fingers.append(1)
                else:
                    fingers.append(0)

            total_fingers = sum(fingers)
            cv2.putText(frame, f'Fingers: {total_fingers}', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    # FPS Hesaplama
    cTime = time.time()
    fps = 1 / (cTime - pTime) if cTime != pTime else 0
    pTime = cTime

    # FPS ekrana yazdırma
    cv2.putText(frame, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

    cv2.imshow("Hand Track", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
