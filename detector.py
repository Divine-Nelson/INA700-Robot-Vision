from ultralytics import YOLO
import time


class ShapeDetector:
    """
    AI object detector for the INA700 ABB vision project.

    Target classes:
        - triangle
        - square
        - circle
    """

    TARGET_CLASSES = {"triangle", "square", "circle"}

    def __init__(self, model_path="models/best.pt", confidence=0.60):
        self.model = YOLO(model_path)
        self.confidence = confidence

    def detect(self, frame):
        """
        Run YOLO inference on a camera frame.

        Returns:
            detections: list of dictionaries
            inference_ms: inference time in milliseconds
        """

        start = time.perf_counter()

        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            verbose=False
        )

        inference_ms = (time.perf_counter() - start) * 1000

        detections = []

        for result in results:
            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                class_name = self.model.names[class_id].lower()

                # Safety: ignore anything outside our assigned shapes.
                if class_name not in self.TARGET_CLASSES:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # YOLO bounding-box center.
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,

                    "bbox": (x1, y1, x2, y2),

                    "bbox_center": (cx, cy),
                })

        return detections, inference_ms