import time
import math
import cv2
import numpy as np
from ultralytics import YOLO
from pypylon import pylon


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    r"C:\Users\diez0001\Documents\ABB_Vision_Training"
    r"\runs\detect\ina700_3class_basler_v3\weights\best.pt"
)

CONFIDENCE = 0.25
IMAGE_SIZE = 640

WINDOW_NAME = "INA700 - YOLO + OpenCV Geometry"

MIN_CONTOUR_AREA = 100


# ============================================================
# GEOMETRY ANALYZER
# ============================================================

class GeometryAnalyzer:

    def __init__(self):
        pass

    # --------------------------------------------------------
    # FIND OBJECT CONTOUR INSIDE YOLO ROI
    # --------------------------------------------------------

    def find_contour(self, frame, bbox):

        x1, y1, x2, y2 = map(int, bbox)

        h, w = frame.shape[:2]

        x1 = max(0, min(x1, w - 1))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h - 1))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return None

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # Otsu threshold.
        #
        # We try both polarities because some physical shapes
        # may be lighter than the background while others are
        # darker.
        _, binary_a = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        binary_b = cv2.bitwise_not(binary_a)

        candidates = []

        for binary in [binary_a, binary_b]:

            contours, _ = cv2.findContours(
                binary,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            for contour in contours:

                area = cv2.contourArea(contour)

                if area < MIN_CONTOUR_AREA:
                    continue

                rx, ry, rw, rh = cv2.boundingRect(contour)

                roi_h, roi_w = binary.shape

                # Reject contours that are basically the
                # complete ROI/background.
                coverage = (
                    area /
                    float(roi_w * roi_h)
                )

                if coverage > 0.90:
                    continue

                candidates.append(
                    (
                        area,
                        contour
                    )
                )

        if not candidates:
            return None

        # Largest useful contour inside the YOLO ROI.
        _, contour = max(
            candidates,
            key=lambda item: item[0]
        )

        # Convert ROI coordinates back into full-image pixels.
        contour = contour.copy()

        contour[:, 0, 0] += x1
        contour[:, 0, 1] += y1

        return contour


    # --------------------------------------------------------
    # CENTROID
    # --------------------------------------------------------

    def centroid(self, contour):

        moments = cv2.moments(contour)

        if abs(moments["m00"]) < 1e-6:
            return None

        cx = (
            moments["m10"] /
            moments["m00"]
        )

        cy = (
            moments["m01"] /
            moments["m00"]
        )

        return cx, cy


    # --------------------------------------------------------
    # TRIANGLE ORIENTATION
    # --------------------------------------------------------

    def triangle_orientation(
        self,
        contour,
        centroid
    ):

        perimeter = cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            0.04 * perimeter,
            True
        )

        # If thresholding produced more than exactly
        # three corners, use the minimum enclosing
        # triangle as a fallback.
        if len(approx) == 3:

            points = approx.reshape(-1, 2)

        else:

            area, triangle = (
                cv2.minEnclosingTriangle(
                    contour
                )
            )

            if triangle is None:
                return None

            points = (
                triangle
                .reshape(-1, 2)
            )

        cx, cy = centroid

        # For a triangle, choose the vertex furthest
        # from the centroid as the directional tip.
        distances = [
            (
                (px - cx) ** 2 +
                (py - cy) ** 2
            )
            for px, py in points
        ]

        tip = points[
            int(np.argmax(distances))
        ]

        dx = tip[0] - cx

        # Image Y grows downward.
        # Negating dy gives conventional Cartesian angle.
        dy = -(tip[1] - cy)

        angle = math.degrees(
            math.atan2(
                dy,
                dx
            )
        )

        angle = angle % 360.0

        return angle


    # --------------------------------------------------------
    # SQUARE ORIENTATION
    # --------------------------------------------------------

    def square_orientation(self, contour):

        rect = cv2.minAreaRect(contour)

        box = cv2.boxPoints(rect)

        box = np.asarray(
            box,
            dtype=np.float32
        )

        # Calculate all four edge directions.
        edges = []

        for i in range(4):

            p1 = box[i]

            p2 = box[
                (i + 1) % 4
            ]

            dx = p2[0] - p1[0]

            dy = -(p2[1] - p1[1])

            length = math.hypot(
                dx,
                dy
            )

            angle = math.degrees(
                math.atan2(
                    dy,
                    dx
                )
            )

            edges.append(
                (
                    length,
                    angle
                )
            )

        # Pick one of the longest edges.
        _, angle = max(
            edges,
            key=lambda item: item[0]
        )

        # Square orientation is physically equivalent
        # every 90 degrees.
        angle = angle % 90.0

        return angle


    # --------------------------------------------------------
    # COMPLETE ANALYSIS
    # --------------------------------------------------------

    def analyze(
        self,
        frame,
        bbox,
        class_name
    ):

        contour = self.find_contour(
            frame,
            bbox
        )

        if contour is None:
            return None

        center = self.centroid(
            contour
        )

        if center is None:
            return None

        angle = None

        if class_name == "triangle":

            angle = (
                self.triangle_orientation(
                    contour,
                    center
                )
            )

        elif class_name == "square":

            angle = (
                self.square_orientation(
                    contour
                )
            )

        elif class_name == "circle":

            # Uniform circle has no meaningful
            # rotational orientation.
            angle = None

        return {
            "contour": contour,
            "center": center,
            "angle": angle,
        }


# ============================================================
# LOAD YOLO
# ============================================================

print("=" * 70)
print("INA700 LIVE YOLO + OPENCV GEOMETRY")
print("=" * 70)

print("\nLoading V3 model...")

model = YOLO(MODEL_PATH)

print("Model loaded.")
print("Classes:", model.names)


# ============================================================
# GEOMETRY
# ============================================================

geometry = GeometryAnalyzer()


# ============================================================
# BASLER CAMERA
# ============================================================

print("\nConnecting to Basler...")

camera = pylon.InstantCamera(
    pylon.TlFactory
    .GetInstance()
    .CreateFirstDevice()
)

camera.Open()

print(
    "Camera:",
    camera.GetDeviceInfo().GetModelName()
)


converter = pylon.ImageFormatConverter()

converter.OutputPixelFormat = (
    pylon.PixelType_BGR8packed
)

converter.OutputBitAlignment = (
    pylon.OutputBitAlignment_MsbAligned
)


camera.StartGrabbing(
    pylon.GrabStrategy_LatestImageOnly
)


print("\nLive geometry started.")
print("Press Q to quit.")
print("=" * 70)


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while camera.IsGrabbing():

        grab = camera.RetrieveResult(
            5000,
            pylon.TimeoutHandling_ThrowException
        )

        if not grab.GrabSucceeded():

            grab.Release()
            continue


        converted = converter.Convert(
            grab
        )

        frame = converted.GetArray()

        grab.Release()


        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        start = time.perf_counter()

        results = model.predict(
            frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False
        )

        inference_ms = (
            time.perf_counter() -
            start
        ) * 1000


        result = results[0]

        display = frame.copy()


        # ----------------------------------------------------
        # PROCESS EACH YOLO DETECTION
        # ----------------------------------------------------

        for box in result.boxes:

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = (
                model.names[class_id]
            )

            x1, y1, x2, y2 = (
                box.xyxy[0]
                .cpu()
                .numpy()
            )


            # ------------------------------------------------
            # YOLO BOX
            # ------------------------------------------------

            cv2.rectangle(
                display,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (255, 255, 255),
                2
            )


            # ------------------------------------------------
            # OPENCV GEOMETRY
            # ------------------------------------------------

            analysis = geometry.analyze(
                frame,
                (x1, y1, x2, y2),
                class_name
            )


            if analysis is None:

                label = (
                    f"{class_name} "
                    f"{confidence:.2f} "
                    f"GEOMETRY FAIL"
                )

                cv2.putText(
                    display,
                    label,
                    (int(x1), max(20, int(y1) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                continue


            contour = analysis[
                "contour"
            ]

            cx, cy = analysis[
                "center"
            ]

            angle = analysis[
                "angle"
            ]


            # ------------------------------------------------
            # DRAW CONTOUR
            # ------------------------------------------------

            cv2.drawContours(
                display,
                [contour],
                -1,
                (255, 255, 255),
                2
            )


            # ------------------------------------------------
            # DRAW CENTROID
            # ------------------------------------------------

            cv2.circle(
                display,
                (
                    int(round(cx)),
                    int(round(cy))
                ),
                7,
                (255, 255, 255),
                -1
            )


            # ------------------------------------------------
            # DRAW ORIENTATION
            # ------------------------------------------------

            if angle is not None:

                length = 80

                radians = math.radians(
                    angle
                )

                end_x = (
                    cx +
                    length *
                    math.cos(radians)
                )

                # Convert Cartesian direction back
                # to image coordinates.
                end_y = (
                    cy -
                    length *
                    math.sin(radians)
                )

                cv2.arrowedLine(
                    display,
                    (
                        int(cx),
                        int(cy)
                    ),
                    (
                        int(end_x),
                        int(end_y)
                    ),
                    (255, 255, 255),
                    3,
                    tipLength=0.25
                )


            # ------------------------------------------------
            # LABEL
            # ------------------------------------------------

            if angle is None:

                angle_text = "angle=N/A"

            else:

                angle_text = (
                    f"angle={angle:.1f}"
                )


            label = (
                f"{class_name} "
                f"{confidence:.2f} "
                f"{angle_text}"
            )


            cv2.putText(
                display,
                label,
                (
                    int(x1),
                    max(
                        25,
                        int(y1) - 10
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )


            # ------------------------------------------------
            # TERMINAL OUTPUT
            # ------------------------------------------------

            print(
                f"{class_name:8s} "
                f"conf={confidence:.2f} "
                f"centroid=({cx:.1f}, {cy:.1f}) "
                f"angle="
                f"{'N/A' if angle is None else f'{angle:.1f}'}"
            )


        # ----------------------------------------------------
        # INFERENCE TIME
        # ----------------------------------------------------

        cv2.putText(
            display,
            f"YOLO inference: {inference_ms:.1f} ms",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL
        )

        cv2.imshow(
            WINDOW_NAME,
            display
        )


        key = (
            cv2.waitKey(1) &
            0xFF
        )

        if key == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

finally:

    print("\nStopping camera...")

    if camera.IsGrabbing():
        camera.StopGrabbing()

    camera.Close()

    cv2.destroyAllWindows()

    print("Camera closed.")
    print("Program finished.")