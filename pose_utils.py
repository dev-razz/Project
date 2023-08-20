import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
)
pose_connections = mp_pose.POSE_CONNECTIONS
mp_drawing = mp.solutions.drawing_utils

joint_mapping = {
    'nose': mp_pose.PoseLandmark.NOSE.value,
    'left_eye_inner': mp_pose.PoseLandmark.LEFT_EYE_INNER.value,
    'left_eye': mp_pose.PoseLandmark.LEFT_EYE.value,
    'left_eye_outer': mp_pose.PoseLandmark.LEFT_EYE_OUTER.value,
    'right_eye_inner': mp_pose.PoseLandmark.RIGHT_EYE_INNER.value,
    'right_eye': mp_pose.PoseLandmark.RIGHT_EYE.value,
    'right_eye_outer': mp_pose.PoseLandmark.RIGHT_EYE_OUTER.value,
    'left_ear': mp_pose.PoseLandmark.LEFT_EAR.value,
    'right_ear': mp_pose.PoseLandmark.RIGHT_EAR.value,
    'mouth_left': mp_pose.PoseLandmark.MOUTH_LEFT.value,
    'mouth_right': mp_pose.PoseLandmark.MOUTH_RIGHT.value,
    'left_shoulder': mp_pose.PoseLandmark.LEFT_SHOULDER.value,
    'right_shoulder': mp_pose.PoseLandmark.RIGHT_SHOULDER.value,
    'left_elbow': mp_pose.PoseLandmark.LEFT_ELBOW.value,
    'right_elbow': mp_pose.PoseLandmark.RIGHT_ELBOW.value,
    'left_wrist': mp_pose.PoseLandmark.LEFT_WRIST.value,
    'right_wrist': mp_pose.PoseLandmark.RIGHT_WRIST.value,
    'left_pinky': mp_pose.PoseLandmark.LEFT_PINKY.value,
    'right_pinky': mp_pose.PoseLandmark.RIGHT_PINKY.value,
    'left_index': mp_pose.PoseLandmark.LEFT_INDEX.value,
    'right_index': mp_pose.PoseLandmark.RIGHT_INDEX.value,
    'left_thumb': mp_pose.PoseLandmark.LEFT_THUMB.value,
    'right_thumb': mp_pose.PoseLandmark.RIGHT_THUMB.value,
    'left_hip': mp_pose.PoseLandmark.LEFT_HIP.value,
    'right_hip': mp_pose.PoseLandmark.RIGHT_HIP.value,
    'left_knee': mp_pose.PoseLandmark.LEFT_KNEE.value,
    'right_knee': mp_pose.PoseLandmark.RIGHT_KNEE.value,
    'left_ankle': mp_pose.PoseLandmark.LEFT_ANKLE.value,
    'right_ankle': mp_pose.PoseLandmark.RIGHT_ANKLE.value,
    'left_heel': mp_pose.PoseLandmark.LEFT_HEEL.value,
    'right_heel': mp_pose.PoseLandmark.RIGHT_HEEL.value,
    'left_foot_index': mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value,
    'right_foot_index': mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value,
}

def calculate_angle(a,b,c):
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = int(np.abs(radians*180.0/np.pi))
    if angle >180.0:
        angle = 360-angle

    return angle

def exercise(joints,joint_angles,image):
    global angles
    landmarks = results.pose_landmarks.landmark
    h, w, _ = image.shape
    joint_ids = []
    for joint in joints:
        joint_ids.append(joint_mapping[joint])
    # Get coordinates
    coordinates = {}
    for joint in joints:
        coordinate = [landmarks[joint_mapping[joint]].x,landmarks[joint_mapping[joint]].y]
        coordinates[joint] = coordinate
    #abs_hip_shoulder = abs(shoulder[0] - hip[0])
    # Calculate angle
    angles = {}
    for joint_angle in joint_angles:
        angle = calculate_angle(coordinates[joint_angle[0]],coordinates[joint_angle[1]],coordinates[joint_angle[2]])
        angles[joint_angle[1]] = angle
        cv2.putText(image, str(angle),tuple(np.multiply(coordinates[joint_angle[1]], [640, 480]).astype(int)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

    # Draw circles on highlighted joints
    for joint in joints:
        landmark = landmarks[joint_mapping[joint]]
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 8, (0, 255, 0), -1)
    # Draw lines between highlighted joints
    for connection in pose_connections:
        joint1, joint2 = connection
        if joint1 in joint_ids and joint2 in joint_ids:
            landmark1 = landmarks[joint1]
            landmark2 = landmarks[joint2]
            pt1 = (int(landmark1.x * w), int(landmark1.y * h))
            pt2 = (int(landmark2.x * w), int(landmark2.y * h))
            cv2.line(image, pt1, pt2, (255, 0, 0), 2)

def tracker(image):
    global results
    results = pose.process(image)
    joints = ["right_hip", "right_shoulder", "right_elbow", "right_wrist"]
    joint_angles = [["right_hip", "right_shoulder", "right_elbow"],["right_shoulder", "right_elbow", "right_wrist"]]
    try:
        exercise(joints, joint_angles,image)
    except Exception as e:
        print(e)
        pass
    return image