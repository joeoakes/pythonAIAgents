import asyncio
import random  # Simulating sensor data
import cv2
import numpy as np

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

            # Simulated object detection (randomly trigger detection)
            if random.random() > 0.7:
                await message_queue.put("CameraAgent: Object detected in frame")

            print("CameraAgent: Captured frame.")

    def release(self):
        """Release the camera resource"""
        self.cap.release()


class MainController:
    """Main Controller listening for notifications"""

    async def listen_for_notifications(self):
        while True:
            message = await message_queue.get()
            print(f"MainController: Received -> {message}")


async def main():
    sonic_agent = SonicAgent()
    camera_agent = CameraAgent()
    controller = MainController()

    # Run agents asynchronously
    tasks = [
        asyncio.create_task(sonic_agent.detect_object()),
        asyncio.create_task(camera_agent.detect_object()),
        asyncio.create_task(controller.listen_for_notifications()),
    ]

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("Shutting down...")
