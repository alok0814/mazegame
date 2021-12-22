import cv2
import mediapipe as mp
import numpy as np
import time
import random
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

device = 0 # camera device number

def getFrameNumber(start:float, fps:int):
    now = time.perf_counter() - start
    frame_now = int(now * 1000 / fps)

    return frame_now

def judgeOpen(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        # Convert the obtained landmark values x and y to the coordinates on the image
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])

        if len(landmark_point) != 0 and len(landmark_point)==21:
            vec1 = (landmark_point[5][0] - landmark_point[6][0], 
            landmark_point[5][1] - landmark_point[6][1], 
            landmark_point[5][2] - landmark_point[6][2])
            vec2 = (landmark_point[7][0] - landmark_point[6][0], 
            landmark_point[7][1] - landmark_point[6][1], 
            landmark_point[7][2] - landmark_point[6][2])
            vec3 = (landmark_point[1][0] - landmark_point[2][0], 
            landmark_point[1][1] - landmark_point[2][1], 
            landmark_point[1][2] - landmark_point[2][2])
            vec4 = (landmark_point[3][0] - landmark_point[2][0], 
            landmark_point[3][1] - landmark_point[3][1], 
            landmark_point[3][2] - landmark_point[3][2])
            vec5 = (landmark_point[9][0] - landmark_point[10][0], 
            landmark_point[9][1] - landmark_point[10][1], 
            landmark_point[9][2] - landmark_point[10][2])
            vec6 = (landmark_point[11][0] - landmark_point[10][0], 
            landmark_point[11][1] - landmark_point[10][1], 
            landmark_point[11][2] - landmark_point[10][2])
            vec7 = (landmark_point[13][0] - landmark_point[14][0], 
            landmark_point[13][1] - landmark_point[14][1], 
            landmark_point[13][2] - landmark_point[14][2])
            vec8 = (landmark_point[15][0] - landmark_point[14][0], 
            landmark_point[15][1] - landmark_point[14][1], 
            landmark_point[15][2] - landmark_point[14][2])
            vec9 = (landmark_point[17][0] - landmark_point[18][0], 
            landmark_point[17][1] - landmark_point[18][1], 
            landmark_point[17][2] - landmark_point[18][2])
            vec10 = (landmark_point[19][0] - landmark_point[18][0], 
            landmark_point[19][1] - landmark_point[18][1], 
            landmark_point[19][2] - landmark_point[18][2])
            finger_num = 0
            if calcAngle(vec1, vec2) > 140:
                finger_num += 1
            if calcAngle(vec3, vec4) > 140:
                finger_num += 1
            if calcAngle(vec5, vec6) > 140:
                finger_num += 1
            if calcAngle(vec7, vec8) > 140:
                finger_num += 1
            if calcAngle(vec9, vec10) > 140:
                finger_num += 1

            if finger_num == 0 :
                cv2.putText(image, '0', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5)
                return 0;
            elif finger_num == 1 :
                cv2.putText(image, '1', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5)
                return 1;
            elif finger_num == 2 :
                cv2.putText(image, '2', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5)
                return 2;
            elif finger_num == 3 :
                cv2.putText(image, '3', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5)
                return 3;
            elif finger_num == 4 :
                cv2.putText(image, '4', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5)
                return 4;
            elif finger_num == 5 :
                cv2.putText(image, '5', (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255),5)
                return 5;

def calcAngle(v1, v2):
    v1_n = np.linalg.norm(v1)
    v2_n = np.linalg.norm(v2)

    cos_theta = np.inner(v1, v2) / (v1_n * v2_n)

    return np.rad2deg(np.arccos(cos_theta))

def MazeGame(image, landmarks,a,b,c,flag):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_point = []
    
    

    for index, landmark in enumerate(landmarks.landmark):
        if landmark.visibility < 0 or landmark.presence < 0:
            continue

        # Convert the obtained landmark values x and y to the coordinates on the image
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y, landmark_z])

    # Draw a circle on index finger and display the coordinate value
    cv2.circle(image, (landmark_point[8][0], landmark_point[8][1]), 7, (0, 0, 0), 3)
    
    if 40 < landmark_point[8][0] < 60 and 40 < landmark_point[8][1] < 60 :
        cv2.circle(image, (landmark_point[8][0], landmark_point[8][1]), 7, (0, 0, 255), 3)
        frame = cv2.putText(image, 'START', (105, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), thickness=10)

    
    if (40 < landmark_point[8][0] < 290 and 40 < landmark_point[8][1] < 60) or (260 < landmark_point[8][0] < 290 and 40 < landmark_point[8][1] < 150) or (40 < landmark_point[8][0] < 290 and 110 < landmark_point[8][1] < 150) or (40 < landmark_point[8][0] < 60 and 40 < landmark_point[8][1] < 440)  or (40 < landmark_point[8][0] < 600 and 410 < landmark_point[8][1] < 440)  or (80 < landmark_point[8][0] < 100 and 170 < landmark_point[8][1] < 410)  or (80 < landmark_point[8][0] < 290 and 170 < landmark_point[8][1] < 190)  or (80 < landmark_point[8][0] < 290 and 260 < landmark_point[8][1] < 280)  or (270 < landmark_point[8][0] < 290 and 170 < landmark_point[8][1] < 280) or (270 < landmark_point[8][0] < 290 and 300 < landmark_point[8][1] < 410) or (80 < landmark_point[8][0] < 290 and 300 < landmark_point[8][1] < 320) or (80 < landmark_point[8][0] < 290 and 390 < landmark_point[8][1] < 410)or (590 < landmark_point[8][0] < 600 and 40 < landmark_point[8][1] < 440) or (310 < landmark_point[8][0] < 600 and 40 < landmark_point[8][1] < 50) or (310 < landmark_point[8][0] < 320 and 40 < landmark_point[8][1] < 410) or (310 < landmark_point[8][0] < 570 and 390 < landmark_point[8][1] < 410) or (340 < landmark_point[8][0] < 600 and 350 < landmark_point[8][1] < 370) or (340 < landmark_point[8][0] < 360 and 70 < landmark_point[8][1] < 370) or (340 < landmark_point[8][0] < 570 and 70 < landmark_point[8][1] < 90) or (550 < landmark_point[8][0] < 570 and 70 < landmark_point[8][1] < 330) or (380 < landmark_point[8][0] < 570 and 310 < landmark_point[8][1] < 330) or (380 < landmark_point[8][0] < 400 and 110 < landmark_point[8][1] < 330) or (380 < landmark_point[8][0] < 530 and 110 < landmark_point[8][1] < 130) or (510 < landmark_point[8][0] < 530 and 110 < landmark_point[8][1] < 290) or (420 < landmark_point[8][0] < 590 and 270 < landmark_point[8][1] < 290) or (420 < landmark_point[8][0] < 440 and 150 < landmark_point[8][1] < 250) or (420 < landmark_point[8][0] < 490 and 150 < landmark_point[8][1] < 170) or (460 < landmark_point[8][0] < 490 and 150 < landmark_point[8][1] < 250):
        cv2.circle(image, (landmark_point[8][0], landmark_point[8][1]), 7, (0, 0, 255), 3)
    else:
        # flag = True
        # cv2.putText(frame, "LIMIT:" + str(flag), (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
        overMap(image)
    
        
    if( 270 < landmark_point[8][0] < 290 and 40 < landmark_point[8][1] < 60):
        # frame = cv2.drawMarker(frame, (280, 50), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=15)
        cv2.putText(image, str(a), (270, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
    if( 270 < landmark_point[8][0] < 290 and 170 < landmark_point[8][1] < 190):
        # frame = cv2.drawMarker(frame, (280, 180), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=15)
        cv2.putText(image, str(b), (270, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
    if( 550 < landmark_point[8][0] < 570 and 390 < landmark_point[8][1] < 410):
        # frame = cv2.drawMarker(frame, (560, 400), (0, 255, 0), markerType=cv2.MARKER_STAR, markerSize=15)
        cv2.putText(image, str(c), (550, 410), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
    if( 445 < landmark_point[8][0] < 485 and 220 < landmark_point[8][1] < 260):
        Ans = judgeOpen(image, landmarks)
        print(a+b+c)
        # Ans = int(input("What's the Answer?"))
        if Ans == a+b+c :
            cv2.putText(image, 'CLEAR', (105, 300), cv2.FONT_HERSHEY_SIMPLEX, 5, (255, 255, 255), thickness=10)
            cv2.putText(image, 'Press esc to finish', (175, 330), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), thickness=2)
    
def drawMap(frame):
    color = (0, 0, 0)
    
    
    frame = cv2.line(frame,(30,30),(610,30),color,3)
    frame = cv2.line(frame,(610,30),(610,460),color,3)
    frame = cv2.line(frame,(300,30),(300,420),color,3)
    frame = cv2.line(frame,(610,460),(30,460),color,3)
    frame = cv2.line(frame,(30,460),(30,30),color,3)

    frame = cv2.line(frame,(70,160),(300,160),color,3)
    frame = cv2.line(frame,(110,290),(300,290),color,3)
    frame = cv2.line(frame,(110,420),(580,420),color,3)
    frame = cv2.line(frame,(70,160),(70,420),color,3)

    frame = cv2.rectangle(frame,(70,70),(260,120),color,3)
    frame = cv2.rectangle(frame,(110,200),(260,250),color,3)
    frame = cv2.rectangle(frame,(110,330),(260,380),color,3)

    frame = cv2.line(frame,(580,420),(580,380),color,3)
    frame = cv2.line(frame,(580,380),(330,380),color,3)
    frame = cv2.line(frame,(330,380),(330,60),color,3)

    frame = cv2.line(frame,(330,60),(580,60),color,3)
    frame = cv2.line(frame,(580,60),(580,340),color,3)
    frame = cv2.line(frame,(580,340),(370,340),color,3)
    frame = cv2.line(frame,(370,340),(370,100),color,3)
    frame = cv2.line(frame,(370,100),(540,100),color,3)
    frame = cv2.line(frame,(540,100),(540,300),color,3)
    frame = cv2.line(frame,(540,300),(410,300),color,3)
    frame = cv2.line(frame,(410,300),(410,140),color,3)
    frame = cv2.line(frame,(410,140),(500,140),color,3)
    frame = cv2.line(frame,(500,140),(500,260),color,3)
    frame = cv2.line(frame,(500,260),(450,260),color,3)
    frame = cv2.line(frame,(450,260),(450,180),color,3)


    frame = cv2.putText(frame, 'S', (40, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
    frame = cv2.putText(frame, 'G', (465, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)

def overMap(frame):
    color = (0, 255, 255)
    
    
    frame = cv2.line(frame,(30,30),(610,30),color,3)
    frame = cv2.line(frame,(610,30),(610,460),color,3)
    frame = cv2.line(frame,(300,30),(300,420),color,3)
    frame = cv2.line(frame,(610,460),(30,460),color,3)
    frame = cv2.line(frame,(30,460),(30,30),color,3)

    frame = cv2.line(frame,(70,160),(300,160),color,3)
    frame = cv2.line(frame,(110,290),(300,290),color,3)
    frame = cv2.line(frame,(110,420),(580,420),color,3)
    frame = cv2.line(frame,(70,160),(70,420),color,3)

    frame = cv2.rectangle(frame,(70,70),(260,120),color,3)
    frame = cv2.rectangle(frame,(110,200),(260,250),color,3)
    frame = cv2.rectangle(frame,(110,330),(260,380),color,3)

    frame = cv2.line(frame,(580,420),(580,380),color,3)
    frame = cv2.line(frame,(580,380),(330,380),color,3)
    frame = cv2.line(frame,(330,380),(330,60),color,3)

    frame = cv2.line(frame,(330,60),(580,60),color,3)
    frame = cv2.line(frame,(580,60),(580,340),color,3)
    frame = cv2.line(frame,(580,340),(370,340),color,3)
    frame = cv2.line(frame,(370,340),(370,100),color,3)
    frame = cv2.line(frame,(370,100),(540,100),color,3)
    frame = cv2.line(frame,(540,100),(540,300),color,3)
    frame = cv2.line(frame,(540,300),(410,300),color,3)
    frame = cv2.line(frame,(410,300),(410,140),color,3)
    frame = cv2.line(frame,(410,140),(500,140),color,3)
    frame = cv2.line(frame,(500,140),(500,260),color,3)
    frame = cv2.line(frame,(500,260),(450,260),color,3)
    frame = cv2.line(frame,(450,260),(450,180),color,3)

def decrease_Timer(timer):
    timer -= 10

def main():
    # For webcam input:
    global device

    cap = cv2.VideoCapture(device)
    fps = cap.get(cv2.CAP_PROP_FPS)
    wt  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    ht  = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print("Size:", ht, "x", wt, "/Fps: ", fps)

    start = time.perf_counter()
    frame_prv = -1
    a = random.randint(0,2)
    b = random.randint(0,2)
    c = random.randint(0,2)
    flag = False
    
    
    cv2.namedWindow('MediaPipe Hands', cv2.WINDOW_NORMAL)
    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7) as hands:
        
        while cap.isOpened():
            frame_now=getFrameNumber(start, fps)
            if frame_now == frame_prv:
                continue
            frame_prv = frame_now
            # print(frame_now)

            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            frame.flags.writeable = False
            results = hands.process(frame)

            # Draw the index finger annotation on the image.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            drawMap(frame)

            timer = int(100-(frame_now/30))

            if flag:
                timer -= 10

            frame = cv2.putText(frame, "LIMIT:" + str(timer), (30, 23), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),2)
            frame = cv2.putText(frame, "Find 3 nums and add.", (220, 23), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0),1)
            if timer <= 1:
                frame = cv2.putText(frame, 'GAMEOVER', (40, 270), cv2.FONT_HERSHEY_SIMPLEX, 3.5, (0,  0, 255), thickness=10)
                if timer <= -1:
                    break;
                    


            if results.multi_hand_landmarks:
                # print(frame_now)
                
                for hand_landmarks in results.multi_hand_landmarks:
                    MazeGame(frame, hand_landmarks,a,b,c,flag)

            cv2.imshow('Maze game', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()

if __name__ == '__main__':
    main()
