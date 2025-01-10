import os
import cv2
from base_camera import BaseCamera
import numpy as np
import robot
import time
import tflite_runtime.interpreter as tflite

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

class Camera(BaseCamera):
    video_source = 0
    modeSelect = 'pose'

    def __init__(self):
        model_path = os.path.join(thisPath, 'models/movenet_single_pose_lightning_ptq.tflite')
        self.interpreter = None
        try:
            print(f"Loading model from: {model_path}")
            self.interpreter = tflite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print("Model loaded successfully")
            print("Input details:", self.input_details)
            print("Output details:", self.output_details)
        except Exception as e:
            print(f"Error loading model: {e}")
            
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

    def process_pose(self, frame):
        if self.interpreter is None:
            print("No interpreter loaded")
            return frame

        try:
            print(f"Frame shape: {frame.shape}")

            # Prepare input image
            input_size = 192
            input_image = cv2.resize(frame, (input_size, input_size))
            input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
            
            # Keep as uint8 (0-255) instead of converting to float32
            input_image = np.expand_dims(input_image, axis=0)
            
            print(f"Processed input shape: {input_image.shape}")
            print(f"Input dtype: {input_image.dtype}")

            # Run model
            self.interpreter.set_tensor(self.input_details[0]['index'], input_image)
            print("Running inference...")
            self.interpreter.invoke()
            print("Inference complete")

            # Get output
            keypoints = self.interpreter.get_tensor(self.output_details[0]['index'])
            print(f"Raw output shape: {keypoints.shape}")
            print(f"First few keypoints: {keypoints[0,0,:3]}")

            # Process keypoints
            height, width = frame.shape[:2]
            keypoints = keypoints[0, 0]  # First person, first instance
            
            # Draw keypoints
            for idx, keypoint in enumerate(keypoints):
                y, x, confidence = keypoint
                print(f"Keypoint {idx}: x={x:.2f}, y={y:.2f}, conf={confidence:.2f}")
                
                if confidence > 0.3:  # Confidence threshold
                    x_px = min(width-1, int(x * width))
                    y_px = min(height-1, int(y * height))
                    # Draw a large red circle
                    cv2.circle(frame, (x_px, y_px), 10, (0, 0, 255), -1)
                    # Draw keypoint label
                    cv2.putText(frame, str(idx), (x_px, y_px), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw skeleton lines
            skeleton_pairs = [
                (KEYPOINT_DICT['left_shoulder'], KEYPOINT_DICT['right_shoulder']),
                (KEYPOINT_DICT['left_shoulder'], KEYPOINT_DICT['left_elbow']),
                (KEYPOINT_DICT['right_shoulder'], KEYPOINT_DICT['right_elbow']),
                (KEYPOINT_DICT['left_elbow'], KEYPOINT_DICT['left_wrist']),
                (KEYPOINT_DICT['right_elbow'], KEYPOINT_DICT['right_wrist']),
                (KEYPOINT_DICT['left_shoulder'], KEYPOINT_DICT['left_hip']),
                (KEYPOINT_DICT['right_shoulder'], KEYPOINT_DICT['right_hip']),
                (KEYPOINT_DICT['left_hip'], KEYPOINT_DICT['right_hip']),
                (KEYPOINT_DICT['left_hip'], KEYPOINT_DICT['left_knee']),
                (KEYPOINT_DICT['right_hip'], KEYPOINT_DICT['right_knee']),
                (KEYPOINT_DICT['left_knee'], KEYPOINT_DICT['left_ankle']),
                (KEYPOINT_DICT['right_knee'], KEYPOINT_DICT['right_ankle'])
            ]

            for pair in skeleton_pairs:
                if (keypoints[pair[0]][2] > 0.3 and 
                    keypoints[pair[1]][2] > 0.3):  # Check confidence
                    
                    start_point = keypoints[pair[0]]
                    end_point = keypoints[pair[1]]
                    
                    start_x = int(start_point[1] * width)
                    start_y = int(start_point[0] * height)
                    end_x = int(end_point[1] * width)
                    end_y = int(end_point[0] * height)
                    
                    cv2.line(frame, (start_x, start_y), (end_x, end_y), 
                            (0, 255, 255), 4)  # Thick yellow lines

            # Check for pose-based control
            right_shoulder_idx = KEYPOINT_DICT['right_shoulder']
            right_wrist_idx = KEYPOINT_DICT['right_wrist']
            
            if (keypoints[right_shoulder_idx][2] > 0.3 and 
                keypoints[right_wrist_idx][2] > 0.3):
                
                shoulder_y = keypoints[right_shoulder_idx][0] * height
                wrist_y = keypoints[right_wrist_idx][0] * height
                
                if wrist_y < shoulder_y:
                    robot.forward()
                    cv2.putText(frame, "Moving Forward", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    robot.stopFB()

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
        frame_count = 0

        while True:
            _, img = camera.read()
            frame_count += 1
            if frame_count % 30 == 0:  # Only print every 30 frames
                print(f"Processing frame {frame_count}")
            
            img = instance.process_pose(img.copy())
            yield cv2.imencode('.jpg', img)[1].tobytes()
