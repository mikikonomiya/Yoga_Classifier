import cv2
import numpy as np
import mediapipe as mp
import joblib
import math 
import pandas as pd
import os
import warnings
warnings.filterwarnings ("ignore", category=UserWarning, module="sklearn")

def calculateAngle(p1, p2, p3):
    (x1, y1, z1 )= p1
    (x2, y2, z2) = p2
    (x3, y3, z3) = p3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def angles_finder(landmarks):
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value])
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value])
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    angle_for_half_moon1 = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    angle_for_half_moon2 = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    left_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value])
    right_hip_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    neck_angle_uk = calculateAngle(landmarks[mp_pose.PoseLandmark.NOSE.value],
                                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                   landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value])
    left_wrist_angle_bk = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    right_wrist_angle_bk = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    angles_list = [left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle,
            left_knee_angle, right_knee_angle, angle_for_half_moon1, angle_for_half_moon2,
             left_hip_angle, right_hip_angle, neck_angle_uk, left_wrist_angle_bk, right_wrist_angle_bk]

    columns = ["left_elbow_angle","right_elbow_angle","left_shoulder_angle","right_shoulder_angle",
           "left_knee_angle","right_knee_angle","angle_for_half_moon1","angle_for_half_moon2",
           "left_hip_angle","right_hip_angle","neck_angle_uk","left_wrist_angle_bk","right_wrist_angle_bk"]

    angles_dataframe = pd.DataFrame([angles_list], columns=columns)
    return angles_dataframe , angles_list
 #--------------------------------same as dataset creator functions---------------------------------------#


ideal_angles = {'Butterfly': [180, 180, 10, 5, 342, 18, 305, 55, 77, 283, 306, 0, 0],
                'Dancer_left': [180, 197, 79, 168, 89, 183, 201, 165, 112, 233, 315, 60, 237],
                'Dancer_right': [168, 175, 164, 84, 181, 267, 201, 152, 139, 240, 242, 135, 335],
                'Downward_dog': [167, 167, 174, 186, 180, 180, 340, 20, 80, 79, 300, 80, 80],
                'Goddess': [166, 193, 91, 87, 241, 118, 262, 99, 112, 249, 296, 132, 229],
                'Half_Moon_left': [174, 189, 84, 107, 181, 185, 248, 102, 171, 286, 306, 125, 324],
                'Half_Moon_right': [171, 188, 114, 82, 178, 180, 258, 112, 68, 185, 312, 36, 231],
                'Tree_left': [304,53,39,25,177,33,5,32,179,248,308,197,147], # arms down
                'Tree_right': [160, 187, 185, 188, 330, 173, 329, 356, 129, 181, 309, 194, 172], # arms up
                'Triangle': [166, 188, 93, 121, 177, 175, 278, 91, 58, 143, 319, 65, 167],
                'Warrior_left': [180, 178, 100, 104, 177, 109, 257, 108, 128, 261, 321, 86, 259],
                'Warrior_right': [179, 180, 109, 100, 247, 185, 255, 102, 103, 227, 276, 102, 269]}

thresholds_good = {'Butterfly':  [30, 30, 10, 10, 10, 10, 30, 30,  15, 15, 70, 360, 360],
                'Dancer_left':  [20, 20, 15, 15, 10, 10, 20, 20,  15, 15, 100, 20, 20],
                'Dancer_right':  [20, 20, 15, 15, 10, 10, 20, 20,  15, 15, 30, 20, 20],
                'Downward_dog':  [15, 15, 20, 20, 15, 15, 20, 20,  15, 15, 340, 20, 20],
                'Goddess':  [10, 10, 15, 15, 10, 10, 20, 20,  15, 15, 10, 20, 20],
                'Half_Moon_left':  [10, 10, 15, 15, 10, 10, 30, 30,  15, 15, 10, 20, 20],
                'Half_Moon_right':  [10, 10, 15, 15, 10, 10, 30, 30,  15, 15, 10, 20, 20],
                'Tree_left': [10, 10, 15, 15, 10, 10, 20, 20,  15, 15, 10, 20, 20],
                'Tree_right':  [10, 10, 15, 15, 10, 10, 20, 20,  15, 15, 10, 20, 20],
                'Triangle':  [10, 10, 15, 15, 10, 10, 20, 20,  15, 60, 360, 360, 360],
                'Warrior_left':  [10, 10, 15, 15, 10, 10, 20, 20,  15, 15, 10, 20, 20],
                'Warrior_right':  [10, 10, 15, 15, 10, 10, 20, 20,  15, 15, 10, 20, 20]} 

thresholds_warn = {'Butterfly': [35, 35, 20, 20, 20, 20, 40, 40, 25, 25, 80, 370, 370],
                'Dancer_left': [30, 30, 25, 25, 20, 20, 30, 30, 25, 25, 110, 30, 30],
                'Dancer_right': [30, 30, 25, 25, 20, 20, 30, 30, 25, 25, 40, 30, 30],
                'Downward_dog': [25, 25, 30, 30, 25, 25, 30, 30, 25, 25, 350, 30, 30],
                'Goddess': [20, 20, 25, 25, 20, 20, 30, 30, 25, 25, 20, 30, 30],
                'Half_Moon_left': [20, 20, 25, 25, 20, 20, 40, 40, 25, 25, 20, 30, 30],
                'Half_Moon_right': [20, 20, 25, 25, 20, 20, 40, 40, 25, 25, 20, 30, 30],
                'Tree_left': [20, 20, 25, 25, 20, 20, 30, 30, 25, 25, 20, 30, 30],
                'Tree_right': [20, 20, 25, 25, 20, 20, 30, 30, 25, 25, 20, 30, 30],
                'Triangle': [20, 20, 25, 25, 20, 20, 30, 30, 25, 70, 370, 370, 370],
                'Warrior_left': [20, 20, 25, 25, 20, 20, 30, 30, 25, 25, 20, 30, 30],
                'Warrior_right': [20, 20, 25, 25, 20, 20, 30, 30, 25, 25, 20, 30, 30]}

def compare_angles(user_angles, ideal_angles, threshold_good, threshold_warn):
    feedback_list = []

    length = len(user_angles)
    for i in range(length):
        error = abs(user_angles[i] - ideal_angles[i])
        if error >180:
            error= 360- error
        if error <= threshold_good[i]:
            feedback = "✅"
        elif error <= threshold_warn[i]:
            feedback = "⚠️"
        else:
            feedback = "❌"

        feedback_list.append(feedback)
    return feedback_list


def load_emoji(filename, size=(24, 24)):
    path = os.path.join("/Users/simaypay/Desktop/Yoga_Classifier-main/feedback_images", filename)
    emoji = cv2.imread(path, cv2.IMREAD_UNCHANGED)  
    if emoji is None:
        raise FileNotFoundError(f"Could not load {filename}")
    return cv2.resize(emoji, size)



def overlay_emoji(frame, emoji_img, x, y, size=24):
    frame_h, frame_w = frame.shape[:2]

    # Resize emoji
    emoji_img = cv2.resize(emoji_img, (size, size), interpolation=cv2.INTER_AREA)
    emoji_h, emoji_w = emoji_img.shape[:2]

    # Adjust x, y if too close to edges
    if x + emoji_w > frame_w:
        x = frame_w - emoji_w
    if y + emoji_h > frame_h:
        y = frame_h - emoji_h
    if x < 0 or y < 0:
        return  # Emoji won't fit

    bgr = emoji_img[:, :, :3]
    alpha = emoji_img[:, :, 3] / 255.0
    
    roi = frame[y:y+emoji_h, x:x+emoji_w]

    for c in range(3):
        roi[:, :, c] = (alpha * bgr[:, :, c] + (1 - alpha) * roi[:, :, c])

def score_calculation(feedback_list, pose):
    # elbow x2, shoulder x2, knee x2, halfmoon x2, hip x2, neck, wrist x2
    weights = {'Butterfly': [4, 4, 10, 10, 8, 8, 4, 4, 9, 9, 4, 4, 4], # importance (%) of each joint in the overall score 
                'Dancer_left': [10, 10, 10, 10, 10, 10, 4, 4, 10, 10, 4, 4, 4],
                'Dancer_right': [10, 10, 10, 10, 10, 10, 4, 4, 10, 10, 4, 4, 4],
                'Downward_dog': [12, 12, 7, 7, 12, 12, 3, 3, 8, 8, 6, 5, 5],
                'Goddess': [9, 9, 8, 8, 15, 15, 3, 3, 10, 10, 6, 2, 2],
                'Half_Moon_left': [9, 9, 8, 8, 9, 9, 8, 8, 8, 8, 6, 5, 5],
                'Half_Moon_right': [9, 9, 8, 8, 9, 9, 8, 8, 8, 8, 6, 5, 5],
                'Tree_left': [10, 10, 10, 10, 10, 10, 4, 4, 10, 10, 4, 4, 4],
                'Tree_right': [10, 10, 10, 10, 10, 10, 4, 4, 10, 10, 4, 4, 4],
                'Triangle': [10, 10, 10, 10, 10, 10, 3, 3, 10, 10, 4, 5, 5],
                'Warrior_left': [10, 10, 10, 10, 10, 10, 4, 4, 10, 10, 4, 4, 4],
                'Warrior_right': [10, 10, 10, 10, 10, 10, 4, 4, 10, 10, 4, 4, 4]}

    feedback_values = {"✅": 1.0, "⚠️": 0.3, "❌": 0.0}

    overall_score = 0
    for i, feedback in enumerate(feedback_list):
        overall_score += feedback_values[feedback] * weights[pose][i]
    return overall_score
    
check_img = load_emoji("/Users/simaypay/Desktop/Yoga_Classifier-main/feedback_images/check.png", size=(24,24)) # Load all 3 emojis
warn_img = load_emoji("/Users/simaypay/Desktop/Yoga_Classifier-main/feedback_images/cross.png", size=(24,24))
cross_img = load_emoji("/Users/simaypay/Desktop/Yoga_Classifier-main/feedback_images/warning.png", size=(24,24))

#-------------------------------feedback system----------------------------------#

mp_pose= mp.solutions.pose
mp_skeleton= mp.solutions.drawing_utils

pose=mp_pose.Pose()

model= joblib.load("mymodel.pkl")

cam= cv2.VideoCapture(0)


joints = [ #these are indexes
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.RIGHT_ELBOW,
                mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.RIGHT_KNEE,
                mp_pose.PoseLandmark.RIGHT_ANKLE, 
                mp_pose.PoseLandmark.LEFT_ANKLE,   
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.RIGHT_HIP,
                mp_pose.PoseLandmark.NOSE,
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.RIGHT_WRIST
            ]
#just variables
frame_no=0
last_prediction = None
last_feedback_list = None


while cam.isOpened():
    success , frame = cam.read()

    
    img_rgb= cv2.cvtColor(frame , cv2.COLOR_BGR2RGB)
    img_copy= frame.copy()
    (h,w,d)= img_copy.shape
    
    landmarks= pose.process(img_rgb)
    
    
    if landmarks.pose_landmarks:
        mp_skeleton.draw_landmarks(img_copy, landmarks.pose_landmarks, mp_pose.POSE_CONNECTIONS)


        frame_no+=1
        coords=landmarks.pose_landmarks.landmark  #the coordinates as a list but normalized (0,1)
        
        standard_coords = [((w*els.x) , (h*els.y), (d* els.z)) for els in coords] #only x,y,z and real coordinates

        if frame_no %20==0 :

            angles_dataframe ,angles= angles_finder(standard_coords) #angles is dataframe to predict 

            #the model prediction
            prediction=model.predict(angles_dataframe)
            

            #feedback down here
            ideal_list = ideal_angles[prediction[0]]
            threshold_good_list = thresholds_good[prediction[0]]
            threshold_warn_list = thresholds_warn[prediction[0]]
            feedback_list = compare_angles(angles, ideal_list, threshold_good_list, threshold_warn_list)


            #cacheing for frame reduction
            last_prediction= prediction
            last_feedback_list= feedback_list
            
        if last_prediction is not None and last_feedback_list is not None:
            for i, joint in enumerate(joints):
                landmark = coords[joint.value]     #coords = landmarks.pose_landmarks.landmark (list)
                if landmark.visibility < 0.5:
                    continue
                x = int(coords[joint.value].x * w)
                y = int(coords[joint.value].y * h)
    
                feedback = last_feedback_list[i]
                
                if feedback == "✅":
                    overlay_emoji(img_copy, check_img, x, y)
                elif feedback == "⚠️":
                    overlay_emoji(img_copy, warn_img, x, y)
                elif feedback == "❌":
                    overlay_emoji(img_copy, cross_img, x, y)
                
           overall_score = score_calculation(last_feedback_list, last_prediction[0])
    
            cv2.putText(img_copy, f"Pose: {last_prediction[0]}", (10, 40), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 2)
            cv2.putText(img_copy, f"Score: {overall_score:.1f}%", (10, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
            if overall_score == 100:
                cv2.putText(img_copy, f"PERFECT!", (10, 160), cv2.FONT_HERSHEY_DUPLEX, 2.3, (30, 150, 0), 3)
            elif overall_score >= 85:
                cv2.putText(img_copy, f"GOOD!", (10, 160), cv2.FONT_HERSHEY_DUPLEX, 2, (200, 0, 0), 3)
    
    cv2.imshow("Yoga Pose Feedback", img_copy)
    
    if cv2.waitKey(1) == ord('q'):
        break


cam.release()
pose.close()
cv2.destroyAllWindows()
