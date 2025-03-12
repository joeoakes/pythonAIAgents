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
import math
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
CAMERA_MATRIX = np.array(config["camera_matrix"])  # Camera Intrinsic Matrix
DIST_COEFFS = np.array(config["dist_coeffs"])  # Distortion Coefficients
TAG_SIZE = config["tag_size"]  # Real-world tag size in meters

# Message queue for inter-agent communication
message_queue = asyncio.Queue()


class SonicAgent:
    """Asynchronous Sonic Sensor Agent"""

    def __init__(self, threshold=10):
        self.threshold = threshold  # Distance threshold in cm

    async def detect_object(self):
        while True:
            await asyncio.sleep(1)  # Simulate sensor reading delay
            distance = random.uniform(5, 20)  # Simulated ultrasonic sensor reading

            if distance < self.threshold:
                await message_queue.put(f"SonicAgent: Object detected at {distance:.2f} cm")

            print(f"SonicAgent: Distance = {distance:.2f} cm")


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
                    tag_center = (int(tag.center[0]), int(tag.center[1]))
                    tag_corners = np.array(tag.corners, dtype=np.float32)

                    # Pose Estimation
                    object_points = np.array([[-TAG_SIZE / 2, TAG_SIZE / 2, 0],
                                              [TAG_SIZE / 2, TAG_SIZE / 2, 0],
                                              [TAG_SIZE / 2, -TAG_SIZE / 2, 0],
                                              [-TAG_SIZE / 2, -TAG_SIZE / 2, 0]], dtype=np.float32)

                    success, rvec, tvec = cv2.solvePnP(object_points, tag_corners, CAMERA_MATRIX, DIST_COEFFS)
                    if success:
                        distance = np.linalg.norm(tvec)
                        rotation_matrix, _ = cv2.Rodrigues(rvec)
                        yaw = math.degrees(math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0]))

                        message = (
                            f"CameraAgent: Detected AprilTag ID {tag.tag_id} | "
                            f"Distance: {distance:.2f}m | "
                            f"Yaw: {yaw:.2f}° | "
                            f"Center: {tag_center}"
                        )
                        await message_queue.put(message)
                        print(message)

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


class LidarAgent(Node):
    """ROS2 Lidar Agent listening to /scan topic."""

    def __init__(self):
        super().__init__('lidar_agent')
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)

    def scan_callback(self, msg):
        min_distance = min(msg.ranges)
        if min_distance < 0.5:
            asyncio.create_task(message_queue.put("LidarAgent: Object detected in scan"))


class LogMessageAgent:
    """Sends logs to a remote MongoDB server."""

    def __init__(self, db_url, db_name, collection_name):
        self.client = pymongo.MongoClient(db_url)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def log_message(self, message):
        log_entry = {"turtlebot": TURTLEBOT_NAME, "message": message}
        self.collection.insert_one(log_entry)
        print(f"LogMessageAgent: Logged message - {message}")


class MoveBotAgent(Node):
    """ROS2 MoveBot Agent that listens for commands and publishes to /cmd_vel."""

    def __init__(self):
        super().__init__('move_bot_agent')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

    def move_forward(self):
        msg = Twist()
        msg.linear.x = 0.2
        self.publisher.publish(msg)
        print("MoveBotAgent: Moving forward")

    def stop(self):
        msg = Twist()
        msg.linear.x = 0.0
        self.publisher.publish(msg)
        print("MoveBotAgent: Stopping")


class HMACAgent:
    """Handles sending and receiving authenticated messages using HMAC."""

    def send_authenticated_message(self, message):
        hmac_digest = hmac.new(SECRET_KEY, message.encode(), hashlib.sha256).hexdigest()
        payload = f"{message}|{hmac_digest}"

        for ip in TURTLEBOT_IPS:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
                    sock.sendto(payload.encode(), (ip, 5005))
                print(f"HMACAgent: Sent message to {ip}")
            except Exception as e:
                print(f"HMACAgent: Failed to send message to {ip}: {e}")

    def receive_authenticated_message(self):
        """Listens for authenticated messages and verifies their integrity."""
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind(("", 5005))
            print("HMACAgent: Listening for messages...")
            while True:
                data, addr = sock.recvfrom(1024)
                message, received_hmac = data.decode().rsplit("|", 1)
                expected_hmac = hmac.new(SECRET_KEY, message.encode(), hashlib.sha256).hexdigest()

                if hmac.compare_digest(received_hmac, expected_hmac):
                    print(f"HMACAgent: Received valid message from {addr}: {message}")
                else:
                    print(f"HMACAgent: WARNING! Invalid HMAC from {addr}")


class MainController:
    """Main Controller listening for notifications"""

    async def listen_for_notifications(self):
        while True:
            message = await message_queue.get()
            print(f"MainController: Received -> {message}")


async def main():
    rclpy.init()

    sonic_agent = SonicAgent()
    camera_agent = CameraAgent()
    lidar_agent = LidarAgent()
    log_agent = LogMessageAgent("mongodb://localhost:27017/", "robot_logs", "messages")
    move_bot_agent = MoveBotAgent()
    hmac_agent = HMACAgent()
    controller = MainController()

    # Example usage of HMACAgent
    hmac_agent.send_authenticated_message("TurtleBot Status Update")

    tasks = [
        asyncio.create_task(sonic_agent.detect_object()),
        asyncio.create_task(camera_agent.detect_object()),
        asyncio.create_task(controller.listen_for_notifications()),
    ]

    await asyncio.gather(*tasks)

    rclpy.spin(lidar_agent)
    rclpy.spin(move_bot_agent)

    rclpy.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")