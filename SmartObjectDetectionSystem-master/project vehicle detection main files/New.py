import threading
import numpy as np
import os
from PyQt6.QtWidgets import (
    QPushButton, QWidget, QLabel, QGroupBox, QVBoxLayout, QFrame, QFileDialog,
    QApplication, QComboBox, QListWidget
)
from PyQt6.QtGui import QIcon, QImage, QPixmap
import cv2 as cv
from ultralytics import YOLO
import sys
from collections import Counter
from sort import Sort  # Custom SORT tracker implementation

# Main Application Window Class
class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Smart Object Detection System")
        self.setWindowIcon(QIcon("face-scan.png"))
        self.setGeometry(0, 0, 1200, 700)

        # Load YOLOv11 trained Model
        self.model = YOLO("best.pt")
        self.webcam_running = False
        self.selected_object = "All Objects"
        self.tracker = Sort()  # SORT Tracker instance
        self.counted_ids = set()  # To prevent double-counting of vehicles
        self.total_vehicle_count = 0

        self.create_widgets()  # GUI setup

        # Load stylesheet
        with open("style.css", "r") as f:
            self.setStyleSheet(f.read())

    def create_widgets(self):
        label = QLabel("Smart Object Detection System", self)
        label.move(470, 10)
        label.setStyleSheet("font-size: 20px; font-weight: bold;")

        group_box = QGroupBox("Detection Options", self)
        group_box.setGeometry(30, 40, 220, 120)

        self.btn_1 = QPushButton("Live Webcam Detection")
        self.btn_1.setIcon(QIcon("webcam.png"))
        self.btn_1.setMinimumHeight(40)
        self.btn_1.clicked.connect(self.clicked_btn_1)

        self.btn_2 = QPushButton("Upload Photo/Video")
        self.btn_2.setIcon(QIcon("file.png"))
        self.btn_2.setMinimumHeight(40)
        self.btn_2.clicked.connect(self.clicked_btn_2)

        vbox = QVBoxLayout()
        vbox.addWidget(self.btn_1)
        vbox.addWidget(self.btn_2)
        group_box.setLayout(vbox)

        self.frame_main = QFrame(self)
        self.frame_main.setGeometry(270, 40, 900, 600)
        self.frame_main.setFrameShape(QFrame.Shape.Box)
        self.video_label = QLabel(self.frame_main)
        self.video_label.setGeometry(0, 0, 900, 600)
        self.video_label.setScaledContents(True)

        filter_frame = QFrame(self)
        filter_frame.setGeometry(30, 180, 220, 460)
        filter_frame.setFrameShape(QFrame.Shape.Box)

        self.combo_box = QComboBox(filter_frame)
        self.combo_box.setGeometry(10, 10, 200, 30)
        self.combo_box.addItem("All Objects")
        for _, name in self.model.names.items():
            self.combo_box.addItem(name)
        self.combo_box.currentTextChanged.connect(self.update_selected_object)

        self.object_list = QListWidget(filter_frame)
        self.object_list.setGeometry(10, 50, 200, 400)

    def update_selected_object(self, text):
        self.selected_object = text

    def clear_video_label(self):
        self.video_label.clear()
        self.object_list.clear()

    def is_vehicle(self, cls_id):
        name = self.model.names.get(cls_id, "").lower()
        return name in ["car", "motorbike", "motorcycle", "bus", "truck", "van", "vehicle"]

    def clicked_btn_1(self):
        if not self.webcam_running:
            self.clear_video_label()
            self.webcam_running = True
            self.btn_1.setText("Stop Detection")
            self.thread = threading.Thread(target=self.run_webcam_detection)
            self.thread.start()
        else:
            self.webcam_running = False
            self.btn_1.setText("Live Webcam Detection")
            self.clear_video_label()

    def clicked_btn_2(self):
        if self.webcam_running:
            self.webcam_running = False
            self.btn_1.setText("Live Webcam Detection")
        self.clear_video_label()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Photo or Video", "", "Media Files (*.png *.jpg *.jpeg *.mp4 *.avi *.mov)"
        )
        if not file_path:
            return
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".jpg", ".jpeg", ".png"]:
            self.detect_on_image(file_path)
        elif ext in [".mp4", ".avi", ".mov"]:
            self.detect_on_video(file_path)

    def detect_on_image(self, file_path):
        frame = cv.imread(file_path)
        object_counter = Counter()
        results = self.model(frame)[0]
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            name = self.model.names.get(cls, "unknown")
            object_counter[name] += 1
            if self.selected_object == "All Objects" or name == self.selected_object:
                cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv.putText(frame, name, (x1, y1 - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        self.update_object_list(object_counter)
        self.show_frame_on_label(frame)

    def detect_on_video(self, file_path):
        if not self.webcam_running:
            self.webcam_running = True
            self.btn_1.setText("Stop Detection")

            def run_video():
                cap = cv.VideoCapture(file_path)
                while self.webcam_running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    object_counter = Counter()
                    results = self.model(frame, stream=True)
                    detections = []
                    for result in results:
                        for box in result.boxes:
                            x1, y1, x2, y2 = map(int, box.xyxy[0])
                            conf = float(box.conf[0])
                            cls_id = int(box.cls[0])
                            name = self.model.names.get(cls_id, "unknown")
                            object_counter[name] += 1
                            if self.is_vehicle(cls_id):
                                detections.append([x1, y1, x2, y2, conf])
                    detections = np.array(detections)
                    if len(detections) > 0:
                        tracked_objects = self.tracker.update(detections)
                        for obj in tracked_objects:
                            x1, y1, x2, y2, track_id = map(int, obj)
                            if track_id not in self.counted_ids:
                                self.counted_ids.add(track_id)
                                self.total_vehicle_count += 1
                            # Retrieve class name for the tracked object
                            name = self.model.names.get(cls_id, "unknown")
                            cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv.putText(frame, f"{name} ID:{track_id}", (x1, y1 - 10),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    self.update_object_list(object_counter)
                    self.show_frame_on_label(frame)
                    if cv.waitKey(25) & 0xFF == ord('q'):
                        break
                cap.release()
                self.webcam_running = False
                self.btn_1.setText("Live Webcam Detection")

            self.thread = threading.Thread(target=run_video)
            self.thread.start()

    def run_webcam_detection(self):
        cam = cv.VideoCapture(0)
        if not cam.isOpened():
            print("Unable to open webcam")
            return
        while self.webcam_running:
            ret, frame = cam.read()
            if not ret:
                break
            object_counter = Counter()
            results = self.model(frame, stream=True)
            detections = []
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    name = self.model.names.get(cls_id, "unknown")
                    object_counter[name] += 1
                    if self.is_vehicle(cls_id):
                        detections.append([x1, y1, x2, y2, conf])
            detections = np.array(detections)
            if len(detections) > 0:
                tracked_objects = self.tracker.update(detections)
                for obj in tracked_objects:
                    x1, y1, x2, y2, track_id = map(int, obj)
                    if track_id not in self.counted_ids:
                        self.counted_ids.add(track_id)
                        self.total_vehicle_count += 1
                    # Retrieve class name for the tracked object
                    name = self.model.names.get(cls_id, "unknown")
                    cv.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv.putText(frame, f"{name} ID:{track_id}", (x1, y1 - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            self.update_object_list(object_counter)
            self.show_frame_on_label(frame)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        cam.release()
        cv.destroyAllWindows()

    def show_frame_on_label(self, frame):
        rgb_image = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap)

    def update_object_list(self, counter):
        self.object_list.clear()
        self.object_list.addItem("Current Frame:")
        for obj, count in counter.items():
            self.object_list.addItem(f"{obj}: {count}")
        self.object_list.addItem(f"\nTotal Vehicles: {self.total_vehicle_count}")

# Launch the application
app = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(app.exec())
