import asyncio
import base64
import json
import cv2
import mediapipe as mp
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi import FastAPI, WebSocket

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize MediaPipe Pose and drawing utils
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a, b, c):
    """Calculate the angle between three points."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

def calculate_distance(p1, p2):
    """Calculate the distance between two points."""
    return np.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

@app.websocket("/ws/{exercise_type}")
async def websocket_endpoint(websocket: WebSocket, exercise_type: str):
    await websocket.accept()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        await websocket.close()
        return
        
    counter = 0
    stage = None
    last_stage = None
    current_stage = None  # Initialize current_stage
    
    with mp_pose.Pose(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as pose:
        try:
            start_time = asyncio.get_event_loop().time()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    current_stage = stage  # Set default current_stage
                    
                    if exercise_type.lower() == "squats":
                        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                               landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
                        
                        angle = calculate_angle(hip, knee, ankle)
                        
                        if angle > 150:
                            current_stage = "standing"
                        elif angle < 90:
                            current_stage = "squat"
                            
                        if current_stage == "standing" and last_stage == "squat":
                            counter += 1
                            
                    elif exercise_type.lower() == "pushups":
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        
                        angle = calculate_angle(shoulder, elbow, wrist)
                        
                        if angle > 160:
                            current_stage = "up"
                        elif angle < 90:
                            current_stage = "down"
                            
                        if current_stage == "up" and last_stage == "down":
                            counter += 1

                    elif exercise_type.lower() == "shoulder_tap":
                        left_hand = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        right_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                        
                        distance = calculate_distance(left_hand, right_shoulder)

                        if distance < 0.1:  
                            current_stage = "touched"
                        else:
                            current_stage = "not_touched"
                            
                        if current_stage == "not_touched" and last_stage == "touched":
                            counter += 1
                            
                    elif exercise_type.lower() == "bicep_curl":
                        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                                   landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                        elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                                 landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                        wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                        
                        angle = calculate_angle(shoulder, elbow, wrist)
                        
                        if angle > 130:
                            current_stage = "down"
                        elif angle < 60:
                            current_stage = "up"
                            
                        if current_stage == "down" and last_stage == "up":
                            counter += 1
                    
                    # Draw pose landmarks
                    mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                    )
                    
                    # Display counter and current stage
                    cv2.putText(image, f'Count: {counter}', (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, f'Stage: {current_stage}', (10, 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Show exercise name popup for first 3 seconds
                current_time = asyncio.get_event_loop().time()
                if current_time - start_time < 3:
                    cv2.putText(image, f'EXERCISE: {exercise_type.upper()}', 
                              (int(image.shape[1]/4), int(image.shape[0]/2)),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
                
                # Encode and send frame
                _, buffer = cv2.imencode('.jpg', image)
                base64_frame = base64.b64encode(buffer).decode('utf-8')
                
                await websocket.send_text(json.dumps({
                    "exercise": exercise_type,
                    "count": counter,
                    "frame": f"data:image/jpeg;base64,{base64_frame}"
                }))
                
                # Update stages
                stage = current_stage
                last_stage = current_stage
                
                # Reduced sleep time for more responsive counter
                await asyncio.sleep(0.016)  # Approximately 60 FPS
                
        except Exception as e:
            print(f"Error in WebSocket connection: {e}")
        finally:
            cap.release()
            await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)