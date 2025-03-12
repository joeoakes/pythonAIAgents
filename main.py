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

# Load configuration from JSON file
with open("config.json", "r") as config_file:
    config = json.load(config_file)
TURTLEBOT_NAME = config["turtlebot_name"]
TURTLEBOT_IPS = config["turtlebot_ips"]
SECRET_KEY = config["secret_key"].encode()

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
    """Asynchronous Camera Agent"""

    def __init__(self, camera_index=0):
        self.cap = cv2.VideoCapture(camera_index)

    async def detect_object(self):
        while True:
            await asyncio.sleep(2)  # Simulate frame processing delay
            ret, frame = self.cap.read()
            if not ret:
                print("CameraAgent: Failed to capture image")
                continue

            if random.random() > 0.7:
                await message_queue.put("CameraAgent: Object detected in frame")

            print("CameraAgent: Captured frame.")

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
