import os
import cv2
from base_camera import BaseCamera
import numpy as np
import robot
import time
import tflite_runtime.interpreter as tflite
import math
from collections import deque
from bark import play_bark

curpath = os.path.realpath(__file__)
thisPath = "/" + os.path.dirname(curpath)

KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

def calculate_angle(point1, point2, point3):
    """Calculate angle between three points"""
    if any(p is None for p in [point1, point2, point3]):
        return None
        
    a = np.array(point1)
    b = np.array(point2)
    c = np.array(point3)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
    angle = np.arccos(cosine_angle)
    
    return np.degrees(angle)

class Camera(BaseCamera):
    video_source = 0
    modeSelect = 'pose'

    def __init__(self):
        model_path = os.path.join(thisPath, 'models/movenet_single_pose_lightning_ptq.tflite')
        self.interpreter = None
        try:
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("Model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            
        # Initialize slouch detection parameters
        self.slouch_window = deque(maxlen=30)  # Store last 30 frames of posture angles
        self.slouch_count = 0
        self.slouch_threshold = 150  # Adjusted to be less sensitive
        self.slouch_trigger_count = 15  # Need 15 frames of slouching to trigger
        
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    def check_side_visibility(self, keypoints, side='right'):
        if side == 'right':
            side_points = [keypoints[KEYPOINT_DICT['right_ear']][2],
                         keypoints[KEYPOINT_DICT['right_shoulder']][2],
                         keypoints[KEYPOINT_DICT['right_hip']][2]]
        else:
            side_points = [keypoints[KEYPOINT_DICT['left_ear']][2],
                         keypoints[KEYPOINT_DICT['left_shoulder']][2],
                         keypoints[KEYPOINT_DICT['left_hip']][2]]
        
        return np.mean(side_points)

    def check_arm_raised(self, keypoints, height, width):
        """Check if arm is raised to control robot movement"""
        right_shoulder = keypoints[KEYPOINT_DICT['right_shoulder']]
        right_wrist = keypoints[KEYPOINT_DICT['right_wrist']]
        
        if right_shoulder[2] > 0.3 and right_wrist[2] > 0.3:
            shoulder_y = right_shoulder[0] * height
            wrist_y = right_wrist[0] * height
            
            if wrist_y < shoulder_y - 30:  # Wrist significantly above shoulder
                return True
        return False

    def process_pose(self, frame):
        if self.interpreter is None:
            return frame

        try:
            # Prepare input image
            input_size = 192
            input_image = cv2.resize(frame, (input_size, input_size))
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            input_image = np.expand_dims(input_image, axis=0)

            # Run inference
            self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
            self.interpreter.invoke()

            # Get keypoints
            keypoints = self.interpreter.get_tensor(self.output_details[0]['index'])
            keypoints = keypoints[0, 0]
            
            # Process keypoints
            height, width = frame.shape[:2]
            processed_keypoints = []
            
            # Draw keypoints
            for idx, keypoint in enumerate(keypoints):
                y, x, confidence = keypoint
                if confidence > 0.3:
                    x_px = min(width-1, int(x * width))
                    y_px = min(height-1, int(y * height))
                    cv2.circle(frame, (x_px, y_px), 5, (0, 0, 255), -1)
                    processed_keypoints.append((x_px, y_px))
                else:
                    processed_keypoints.append(None)

            # Check for arm raise to control robot
            if self.check_arm_raised(keypoints, height, width):
                cv2.putText(frame, "ARM RAISED - Moving Forward", 
                          (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 
                          1, (0, 255, 0), 2)
                robot.forward()
            else:
                robot.stopFB()

            # Check which side is more visible for posture detection
            right_visibility = self.check_side_visibility(keypoints, 'right')
            left_visibility = self.check_side_visibility(keypoints, 'left')
            
            ear = shoulder = hip = None
            side_text = ""
            
            if right_visibility > left_visibility and right_visibility > 0.3:
                ear = processed_keypoints[KEYPOINT_DICT['right_ear']]
                shoulder = processed_keypoints[KEYPOINT_DICT['right_shoulder']]
                hip = processed_keypoints[KEYPOINT_DICT['right_hip']]
                side_text = "Right"
            elif left_visibility > 0.3:
                ear = processed_keypoints[KEYPOINT_DICT['left_ear']]
                shoulder = processed_keypoints[KEYPOINT_DICT['left_shoulder']]
                hip = processed_keypoints[KEYPOINT_DICT['left_hip']]
                side_text = "Left"

            # Process posture if we have valid points
            if all(point is not None for point in [ear, shoulder, hip]):
                posture_angle = calculate_angle(ear, shoulder, hip)
                
                if posture_angle is not None:
                    # Add current angle to window
                    self.slouch_window.append(posture_angle < self.slouch_threshold)
                    
                    # Calculate percentage of slouched frames
                    slouch_percentage = sum(self.slouch_window) / len(self.slouch_window) * 100
                    
                    # Draw the angle and side being tracked
                    cv2.putText(frame, f"{side_text} Side Angle: {posture_angle:.1f}", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                              1, (255, 255, 255), 2)
                    
                    # Draw posture indicator
                    if len(self.slouch_window) >= self.slouch_trigger_count and \
                       slouch_percentage > 80:  # 80% of recent frames show slouching
                        cv2.putText(frame, "SLOUCHING DETECTED!", 
                                  (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1, (0, 0, 255), 2)
                        play_bark()
                    
                    # Draw posture lines
                    cv2.line(frame, ear, shoulder, (0, 255, 0), 2)
                    cv2.line(frame, shoulder, hip, (0, 255, 0), 2)

            return frame

        except Exception as e:
            print(f"Error in process_pose: {e}")
            import traceback
            traceback.print_exc()
            return frame

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def frames():
        camera = cv2.VideoCapture(Camera.video_source)
        camera.set(3, 640)
        camera.set(4, 480)
        
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        instance = Camera()

        while True:
            _, img = camera.read()
            img = instance.process_pose(img.copy())
            yield cv2.imencode('.jpg', img)[1].tobytes()