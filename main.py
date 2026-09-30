import cv2

from detector import ShapeDetector
from geometry import GeometryAnalyzer
from camera import WebcamCamera, BaslerCamera


# ============================================================
# CONFIGURATION
# ============================================================

USE_BASLER = True

MODEL_PATH = "models/best.pt"

CONFIDENCE_THRESHOLD = 0.60


# ============================================================
# DRAW DETECTION
# ============================================================

def draw_detection(frame, detection, geometry):

    x1, y1, x2, y2 = detection["bbox"]

    name = detection["class_name"]
    confidence = detection["confidence"]

    # Bounding box
    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = (
        f"{name.upper()} "
        f"{confidence:.2f}"
    )

    cv2.putText(
        frame,
        label,
        (x1, max(25, y1 - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    if geometry is None:
        return

    cx, cy = geometry["center"]

    # Precise OpenCV centroid
    cv2.circle(
        frame,
        (cx, cy),
        6,
        (0, 0, 255),
        -1
    )

    cv2.drawContours(
        frame,
        [geometry["contour"]],
        -1,
        (255, 0, 0),
        2
    )

    angle = geometry["angle"]

    if angle is None:
        angle_text = "Angle: N/A"
    else:
        angle_text = f"Angle: {angle:.1f} deg"

    cv2.putText(
        frame,
        f"Center: ({cx}, {cy})",
        (x1, y2 + 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        angle_text,
        (x1, y2 + 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 0, 255),
        2
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print("INA700 - ABB AI Vision System")
    print("=" * 60)

    # --------------------------------------------------------
    # Camera
    # --------------------------------------------------------

    if USE_BASLER:
        print("[INFO] Using Basler camera")
        camera = BaslerCamera()

    else:
        print("[INFO] Using webcam")
        camera = WebcamCamera()

    # --------------------------------------------------------
    # AI
    # --------------------------------------------------------

    print("[INFO] Loading YOLO model...")

    detector = ShapeDetector(
        model_path=MODEL_PATH,
        confidence=CONFIDENCE_THRESHOLD
    )

    print("[INFO] Model loaded")
    print("[INFO] Press Q to quit")

    try:

        while True:

            frame = camera.get_frame()

            if frame is None:
                continue

            # -----------------------------------------------
            # AI inference
            # -----------------------------------------------

            detections, inference_ms = detector.detect(frame)

            # Requirement: print inference time.
            print(
                f"Inference: {inference_ms:.1f} ms | "
                f"Objects: {len(detections)}"
            )

            # -----------------------------------------------
            # Geometry
            # -----------------------------------------------

            for detection in detections:

                geometry = GeometryAnalyzer.analyze(
                    frame,
                    detection
                )

                draw_detection(
                    frame,
                    detection,
                    geometry
                )

            # -----------------------------------------------
            # Display inference time
            # -----------------------------------------------

            cv2.putText(
                frame,
                f"Inference: {inference_ms:.1f} ms",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.imshow(
                "ABB AI Vision System",
                frame
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    finally:

        camera.release()

        cv2.destroyAllWindows()

        print("[INFO] Vision system stopped.")


if __name__ == "__main__":
    main()