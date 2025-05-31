import math
import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize mediapipe pose class and drawing utils
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

# Calculate angle between 3 points function
def calculateAngle(p1, p2, p3):
    (x1, y1, z1 )= p1
    (x2, y2, z2) = p2
    (x3, y3, z3) = p3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    return angle

def angles_finder(landmarks):
    left_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value ],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value ])
    right_elbow_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value ],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value])   
    left_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value])
    right_shoulder_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value ])
    left_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    right_knee_angle = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value ],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    angle_for_half_moon1 = calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value],
                                                 landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value ],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    angle_for_half_moon2 = calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value ],
                                                 landmarks[mp_pose.PoseLandmark.LEFT_HIP.value ],
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
    return [left_elbow_angle, right_elbow_angle, left_shoulder_angle, right_shoulder_angle,
            left_knee_angle, right_knee_angle, angle_for_half_moon1, angle_for_half_moon2,
        left_hip_angle, right_hip_angle, neck_angle_uk, left_wrist_angle_bk, right_wrist_angle_bk]

def detect_pose(image, pose):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    landmarks = pose.process(image_rgb)
    if landmarks.pose_landmarks:
        h, w, d = image.shape
        standard_coords = [(int(lm.x * w), int(lm.y * h), lm.z * w) for lm in landmarks.pose_landmarks.landmark]
        
        
        return standard_coords
    return None

# Folder containing images
path = '/Users/simaypay/Desktop/Yoga_Classifier-main/TRAIN'

columns = ["Label","left_elbow_angle","right_elbow_angle","left_shoulder_angle","right_shoulder_angle",
           "left_knee_angle","right_knee_angle","angle_for_half_moon1","angle_for_half_moon2",
           "left_hip_angle","right_hip_angle","neck_angle_uk","left_wrist_angle_bk","right_wrist_angle_bk"]

table = []

for foldername in os.listdir(path):
    folder_path = os.path.join(path, foldername)
    if os.path.isdir(folder_path):
        
        for filename in os.listdir(folder_path):
            
            
            end = os.path.splitext(filename.lower())[1]
            if end in ['.jpg', '.jpeg', '.png']:
        

                img_path = os.path.join(folder_path, filename)
                
                image = cv2.imread(img_path)
                
                landmarks = detect_pose(image, pose)
                if landmarks is not None:
                    angle_list = angles_finder(landmarks)
                    row = [foldername] + angle_list
                    table.append(row)
                else:
                    pass




dataframe = pd.DataFrame(table, columns=columns)
dataframe.to_csv('dataset_images.csv', index=False)
pose.close()
print(dataframe.head())

