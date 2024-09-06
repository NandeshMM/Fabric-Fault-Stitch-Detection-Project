import cv2
import threading
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO
import numpy as np

class YOLOVideoCapture:
    def __init__(self, model_path, img_size=320, frame_skip=2):
        self.model = YOLO(model_path)
        self.img_size = img_size
        self.frame_skip = frame_skip
        self.frame_count = 0
        self.frame = None
        self.running = False
        self.cap = cv2.VideoCapture(0)

        if not self.cap.isOpened():
            raise ValueError("Error: Could not open camera.")

        self.root = tk.Tk()
        self.root.title("YOLO Video Capture")

        # Initialize GUI components
        self.start_button = tk.Button(self.root, text="Start", command=self.start_capture)
        self.start_button.pack()

        self.stop_button = tk.Button(self.root, text="Stop", command=self.stop)
        self.stop_button.pack()

    def start_capture(self):
        if not self.running:
            self.running = True
            self.capture_thread = threading.Thread(target=self.capture_frames)
            self.capture_thread.start()

            self.display_thread = threading.Thread(target=self.run)
            self.display_thread.start()

    def capture_frames(self):
        while self.running:
            ret, new_frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame.")
                self.running = False
                break
            self.frame = new_frame

    def run(self):
        # Create a named window and set it to fullscreen
        cv2.namedWindow('YOLO Detection', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('YOLO Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while self.running:
            if self.frame is None:
                continue

            self.frame_count += 1
            if self.frame_count % self.frame_skip != 0:
                continue

            # Predict on the frame
            results = self.model.predict(source=self.frame, imgsz=self.img_size, conf=0.29, show=False)

            # Create a copy of the frame for annotation
            annotated_frame = self.frame.copy()

            # Draw bounding boxes on the frame
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    confidence = box.conf[0]  # Confidence score
                    class_id = int(box.cls)
                    label = self.model.names[class_id]  # Original label

                    # Convert labels
                    if label.lower() == 'ss':
                        label = 'skip stitch'
                        color = (0, 0, 255)  # Red color for 'skip stitch'
                    elif label.lower() == 'ls':
                        label = 'loose stitch'
                        color = (255, 0, 0)  # Blue color for 'loose stitch'
                    else:
                        color = (255, 255, 255)  # Default color (white) for other classes

                    # Draw the bounding box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Display the annotated frame in fullscreen
            cv2.imshow('YOLO Detection', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

    def stop(self):
        self.running = False
        if self.capture_thread.is_alive():
            self.capture_thread.join()
        if self.display_thread.is_alive():
            self.display_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()
        self.root.quit()

    def start_gui(self):
        self.root.mainloop()

if __name__ == "__main__":
    yolo_video = YOLOVideoCapture(model_path='best4.pt')
    yolo_video.start_gui()
