# Main Controller: Listens for messages from both agents and takes action when notified.
# Sonic Agent: Runs an async loop to read the sensor and sends notifications when an object is detected.
# Camera Agent: Captures frames asynchronously and processes them for object detection.
# LidarAgent: Listens to the /scan topic and detects objects based on distance.
# LogMessageAgent: Logs messages to a remote MongoDB database.
# MoveBotAgent: Uses the /cmd_vel topic to control the Turtlebot’s movement.
# asyncio keeps everything in a single event loop.
# pip install opencv-python
# Developed using Python 3.11.9

import asyncio
import random  # Simulating sensor data
import cv2
import numpy as np
import json
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import pymongo
import hmac
import hashlib
import socket
import apriltag

yolo_model = None  # Placeholder for YOLO model loading

# Load configuration from JSON file
with open("config.json", "r") as config_file:
    config = json.load(config_file)
TURTLEBOT_NAME = config["turtlebot_name"]
TURTLEBOT_IPS = config["turtlebot_ips"]
SECRET_KEY = config["secret_key"].encode()
APRILTAG_IDS = config["apriltag_ids"]
TRAFFIC_CONE_LABELS = config["traffic_cone_labels"]

# Message queue for inter-agent communication
message_queue = asyncio.Queue()


class CameraAgent:
    """Asynchronous Camera Agent with AprilTag and YOLO Detection"""

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)
        self.detector = apriltag.Detector()

    async def detect_object(self):
        while True:
            await asyncio.sleep(2)  # Simulate frame processing delay
            ret, frame = self.cap.read()
            if not ret:
                print("CameraAgent: Failed to capture image")
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            tags = self.detector.detect(gray)
            for tag in tags:
                if tag.tag_id in APRILTAG_IDS:
                    await message_queue.put(f"CameraAgent: Detected AprilTag ID {tag.tag_id}")
                    print(f"CameraAgent: Detected AprilTag ID {tag.tag_id}")

            if yolo_model:
                results = yolo_model(frame)
                for detection in results.xyxy[0]:
                    label = int(detection[-1])
                    if label in TRAFFIC_CONE_LABELS:
                        await message_queue.put(f"CameraAgent: Traffic cone detected! Label: {label}")
                        print(f"CameraAgent: Traffic cone detected! Label: {label}")

            print("CameraAgent: Processed frame.")

    def release(self):
        """Release the camera resource"""
        self.cap.release()


async def main():
    rclpy.init()

    camera_agent = CameraAgent()

    tasks = [
        asyncio.create_task(camera_agent.detect_object()),
    ]

    await asyncio.gather(*tasks)

    rclpy.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")
