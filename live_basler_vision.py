# -*- coding: utf-8 -*-
"""
INA700 - Complete Live Vision Prototype

Pipeline:
    Basler camera
        -> YOLO11n V3 shape detection
        -> OpenCV contour refinement
        -> centroid
        -> orientation
        -> HSV colour classification
        -> ArUco ID 21 detection
        -> homography
        -> pixel coordinates -> marker coordinates in millimetres

Target shapes:
    0 = square
    1 = circle
    2 = triangle

IMPORTANT:
The X/Y values produced here are MARKER-RELATIVE millimetres.
They are NOT ABB/Dobot robot coordinates yet.
"""

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
    r"\runs\detect\ina700_3class_basler_v3"
    r"\weights\best.pt"
)

CONFIDENCE = 0.25
IMAGE_SIZE = 640

CLASS_NAMES = {
    0: "square",
    1: "circle",
    2: "triangle",
}


# ============================================================
# ARUCO CONFIGURATION
# ============================================================

ARUCO_ID = 21

ARUCO_SIZE_MM = 100.0

ARUCO_DICT = cv2.aruco.getPredefinedDictionary(
    cv2.aruco.DICT_5X5_100
)

ARUCO_PARAMETERS = cv2.aruco.DetectorParameters()

ARUCO_DETECTOR = cv2.aruco.ArucoDetector(
    ARUCO_DICT,
    ARUCO_PARAMETERS
)


# ============================================================
# COLOUR DEFINITIONS
# ============================================================
#
# OpenCV HSV:
# H = 0..179
# S = 0..255
# V = 0..255
#
# These are starting values.
# We will tune them using the actual Basler image if necessary.
# ============================================================

COLOR_RANGES = {

    # Red wraps around the OpenCV hue scale
    "red": [
        ((0, 70, 50), (12, 255, 255)),
        ((165, 70, 50), (179, 255, 255)),
    ],

    "orange": [
        ((13, 80, 60), (25, 255, 255)),
    ],

    "yellow": [
        ((26, 70, 70), (35, 255, 255)),
    ],

    "green": [
        ((36, 50, 35), (89, 255, 255)),
    ],

    "blue": [
        ((90, 50, 35), (135, 255, 255)),
    ],

    "white":[
        ((0,0,168), (172,111,255)),
    ],
}

# ============================================================
# BASLER CAMERA
# ============================================================

class BaslerCamera:

    def __init__(self):

        self.camera = pylon.InstantCamera(
            pylon.TlFactory.GetInstance().CreateFirstDevice()
        )

        self.camera.Open()

        print(
            "Basler camera:",
            self.camera.GetDeviceInfo().GetModelName()
        )

        self.converter = pylon.ImageFormatConverter()

        self.converter.OutputPixelFormat = (
            pylon.PixelType_BGR8packed
        )

        self.converter.OutputBitAlignment = (
            pylon.OutputBitAlignment_MsbAligned
        )

        self.camera.StartGrabbing(
            pylon.GrabStrategy_LatestImageOnly
        )


    def read(self):

        grab = self.camera.RetrieveResult(
            5000,
            pylon.TimeoutHandling_ThrowException
        )

        if not grab.GrabSucceeded():
            grab.Release()
            return None

        image = self.converter.Convert(grab)

        frame = image.GetArray().copy()

        grab.Release()

        return frame


    def close(self):

        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()

        self.camera.Close()


# ============================================================
# ARUCO CALIBRATION
# ============================================================

class ArucoCalibration:

    def __init__(self):

        self.H = None

        self.pixel_corners = None


    def detect_and_calibrate(self, frame):

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        corners, ids, rejected = (
            ARUCO_DETECTOR.detectMarkers(gray)
        )

        if ids is None:
            return False

        ids = ids.flatten()

        for marker_corners, marker_id in zip(
            corners,
            ids
        ):

            if marker_id != ARUCO_ID:
                continue

            pts = marker_corners.reshape(4, 2).astype(
                np.float32
            )

            self.pixel_corners = pts

            # ------------------------------------------------
            # IMPORTANT
            #
            # ArUco returns canonical corners:
            #
            # 0 -> 1
            #      |
            # 3 <- 2
            #
            # These coordinates are marker-local.
            # ------------------------------------------------

            physical_points = np.array(
                [
                    [0.0, 0.0],
                    [ARUCO_SIZE_MM, 0.0],
                    [ARUCO_SIZE_MM, ARUCO_SIZE_MM],
                    [0.0, ARUCO_SIZE_MM],
                ],
                dtype=np.float32,
            )

            self.H = cv2.getPerspectiveTransform(
                pts,
                physical_points
            )

            return True

        return False


    def pixel_to_mm(self, u, v):

        if self.H is None:
            return None

        pixel = np.array(
            [[[u, v]]],
            dtype=np.float32
        )

        physical = cv2.perspectiveTransform(
            pixel,
            self.H
        )

        x_mm, y_mm = physical[0, 0]

        return float(x_mm), float(y_mm)


    def draw(self, frame):

        if self.pixel_corners is None:
            return

        pts = self.pixel_corners.astype(int)

        for i, point in enumerate(pts):

            x, y = point

            cv2.circle(
                frame,
                (x, y),
                8,
                (255, 255, 255),
                -1
            )

            cv2.putText(
                frame,
                str(i),
                (x + 12, y - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        cv2.polylines(
            frame,
            [pts],
            True,
            (255, 255, 255),
            2
        )


# ============================================================
# CONTOUR / GEOMETRY
# ============================================================

def find_object_contour(frame, box):

    x1, y1, x2, y2 = map(int, box)

    height, width = frame.shape[:2]

    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(width - 1, x2)
    y2 = min(height - 1, y2)

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

    # Try both threshold polarities

    _, binary1 = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    _, binary2 = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )

    candidates = []

    roi_area = roi.shape[0] * roi.shape[1]

    for binary in (binary1, binary2):

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)

            if area < 100:
                continue

            # Reject contour that is basically the entire ROI

            if area > 0.90 * roi_area:
                continue

            candidates.append(
                (area, contour)
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: item[0],
        reverse=True
    )

    contour = candidates[0][1]

    # Convert ROI contour coordinates
    # back into full-image coordinates.

    contour = contour.copy()

    contour[:, 0, 0] += x1
    contour[:, 0, 1] += y1

    return contour


# ============================================================
# CENTROID
# ============================================================

def contour_centroid(contour):

    moments = cv2.moments(contour)

    if abs(moments["m00"]) < 1e-6:
        return None

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    return cx, cy


# ============================================================
# TRIANGLE ORIENTATION
# ============================================================

def triangle_orientation(contour, center):

    perimeter = cv2.arcLength(
        contour,
        True
    )

    polygon = cv2.approxPolyDP(
        contour,
        0.04 * perimeter,
        True
    )

    if len(polygon) == 3:

        vertices = polygon.reshape(3, 2)

    else:

        area, triangle = cv2.minEnclosingTriangle(
            contour
        )

        if triangle is None:
            return None

        vertices = triangle.reshape(3, 2)


    cx, cy = center

    distances = []

    for vertex in vertices:

        vx, vy = vertex

        distance = (
            (vx - cx) ** 2
            + (vy - cy) ** 2
        )

        distances.append(distance)


    tip = vertices[
        np.argmax(distances)
    ]

    tx, ty = tip

    dx = tx - cx

    # Image Y points downward.
    # Negate to obtain Cartesian orientation.
    dy = -(ty - cy)

    angle = math.degrees(
        math.atan2(dy, dx)
    )

    angle %= 360.0

    return angle, (int(tx), int(ty))


# ============================================================
# SQUARE ORIENTATION
# ============================================================

def square_orientation(contour):

    rect = cv2.minAreaRect(contour)

    box = cv2.boxPoints(rect)

    box = np.asarray(
        box,
        dtype=np.float32
    )

    longest_length = -1

    longest_vector = None

    for i in range(4):

        p1 = box[i]

        p2 = box[(i + 1) % 4]

        vector = p2 - p1

        length = np.linalg.norm(vector)

        if length > longest_length:

            longest_length = length

            longest_vector = vector


    dx = longest_vector[0]

    dy = -longest_vector[1]

    angle = math.degrees(
        math.atan2(dy, dx)
    )

    # Square orientation is equivalent every 90 degrees.

    angle %= 90.0

    return angle


# ============================================================
# COLOUR DETECTION
# ============================================================

# ============================================================
# COLOUR DETECTION
# ============================================================

def detect_color(roi, contour):
    """
    Detect the dominant colour inside the detected object's contour.

    IMPORTANT:
    - 'roi' is the cropped YOLO bounding-box image.
    - 'contour' was extracted from that SAME ROI.
    - Therefore the contour coordinates and image coordinates match.

    Returns:
        color_name : str
        score      : float
    """

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if roi is None or roi.size == 0:
        return "unknown", 0.0

    if contour is None or len(contour) == 0:
        return "unknown", 0.0


    # --------------------------------------------------------
    # Convert ROI from BGR to HSV
    # --------------------------------------------------------

    hsv = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2HSV
    )


    # --------------------------------------------------------
    # Create a mask from the object's contour
    # --------------------------------------------------------

    mask = np.zeros(
        roi.shape[:2],
        dtype=np.uint8
    )

    cv2.drawContours(
        mask,
        [contour],
        contourIdx=-1,
        color=255,
        thickness=-1
    )


    # --------------------------------------------------------
    # Erode mask slightly
    #
    # This removes pixels near the object's boundary.
    # Those pixels may contain:
    # - background
    # - shadows
    # - contour edges
    # - reflections
    # --------------------------------------------------------

    kernel = np.ones(
        (5, 5),
        dtype=np.uint8
    )

    inner_mask = cv2.erode(
        mask,
        kernel,
        iterations=2
    )


    # If erosion removed too much of a small object,
    # fall back to the original contour mask.

    if cv2.countNonZero(inner_mask) < 50:
        inner_mask = mask


    # --------------------------------------------------------
    # Extract HSV pixels belonging ONLY to the object
    # --------------------------------------------------------

    pixels = hsv[inner_mask > 0]

    if len(pixels) == 0:
        return "unknown", 0.0


    H = pixels[:, 0].astype(np.float32)
    S = pixels[:, 1].astype(np.float32)
    V = pixels[:, 2].astype(np.float32)


    # --------------------------------------------------------
    # Diagnostic HSV information
    # --------------------------------------------------------

    median_h = float(np.median(H))
    median_s = float(np.median(S))
    median_v = float(np.median(V))

    mean_h = float(np.mean(H))
    mean_s = float(np.mean(S))
    mean_v = float(np.mean(V))

    print(
        f"  HSV median   : "
        f"H={median_h:.1f}, "
        f"S={median_s:.1f}, "
        f"V={median_v:.1f}"
    )

    print(
        f"  HSV mean     : "
        f"H={mean_h:.1f}, "
        f"S={mean_s:.1f}, "
        f"V={mean_v:.1f}"
    )


    # ========================================================
    # BLACK DETECTION
    # ========================================================
    #
    # Black has very low brightness (V).
    # Hue is not useful for black.
    #

    black_pixels = V < 80

    black_ratio = (
        np.count_nonzero(black_pixels)
        / len(V)
    )


    # ========================================================
    # WHITE DETECTION
    # ========================================================
    #
    # White normally has:
    #
    #   low saturation
    #   high brightness
    #

    white_pixels = (
        (S < 65)
        &
        (V > 140)
    )

    white_ratio = (
        np.count_nonzero(white_pixels)
        / len(V)
    )


    print(
        f"  Black ratio  : {black_ratio:.3f}"
    )

    print(
        f"  White ratio  : {white_ratio:.3f}"
    )


    # --------------------------------------------------------
    # Check achromatic colours first
    # --------------------------------------------------------

    if black_ratio >= 0.30:

        print(
            f"  Colour result: BLACK "
            f"({black_ratio:.3f})"
        )

        return "black", float(black_ratio)


    if white_ratio >= 0.30:

        print(
            f"  Colour result: WHITE "
            f"({white_ratio:.3f})"
        )

        return "white", float(white_ratio)


    # ========================================================
    # CHROMATIC COLOURS
    # ========================================================
    #
    # For actual coloured pixels, reject:
    #
    # - very low saturation pixels
    # - very dark pixels
    #
    # This prevents shadows/highlights from changing the hue.
    #

    chromatic_mask = (
        (S > 60)
        &
        (V > 50)
    )


    chromatic_count = np.count_nonzero(
        chromatic_mask
    )


    if chromatic_count < 20:

        print(
            "  Colour result: UNKNOWN "
            "(not enough chromatic pixels)"
        )

        return "unknown", 0.0


    chromatic_h = H[
        chromatic_mask
    ]


    # --------------------------------------------------------
    # Median hue is more resistant to noise than mean hue
    # --------------------------------------------------------

    hue = float(
        np.median(chromatic_h)
    )


    chromatic_ratio = (
        chromatic_count
        / len(V)
    )


    print(
        f"  Colour hue   : {hue:.1f}"
    )

    print(
        f"  Colour pixels: {chromatic_ratio:.3f}"
    )


    # ========================================================
    # HUE CLASSIFICATION
    #
    # OpenCV hue range:
    #
    # 0 - 179
    #
    # Red wraps around both ends of the scale.
    # ========================================================

    if hue <= 10 or hue >= 170:

        color = "red"


    elif 10 < hue <= 25:

        color = "orange"


    elif 25 < hue <= 35:

        color = "yellow"


    elif 35 < hue <= 89:

        color = "green"


    elif 89 < hue <= 135:

        color = "blue"


    else:

        color = "unknown"


    print(
        f"  Colour result: "
        f"{color.upper()} "
        f"({chromatic_ratio:.3f})"
    )


    return color, float(chromatic_ratio)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("INA700 LIVE VISION")
    print("=" * 65)

    print("Loading YOLO V3...")

    model = YOLO(MODEL_PATH)

    print("Model loaded.")

    camera = BaslerCamera()

    calibration = ArucoCalibration()

    print()
    print("Target ArUco:")
    print(f"DICT_5X5_100 ID {ARUCO_ID}")
    print(f"Physical size: {ARUCO_SIZE_MM:.1f} mm")
    print()
    print("Press Q to quit.")
    print()


    try:

        while True:

            frame = camera.read()

            if frame is None:
                continue


            # =================================================
            # ARUCO
            # =================================================

            aruco_found = (
                calibration.detect_and_calibrate(
                    frame
                )
            )

            calibration.draw(frame)


            # =================================================
            # YOLO
            # =================================================

            start = time.perf_counter()

            results = model.predict(
                frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                device="cpu",
                verbose=False
            )

            inference_ms = (
                time.perf_counter() - start
            ) * 1000.0


            result = results[0]


            print("\n" + "=" * 65)

            if aruco_found:
                print(
                    f"ArUco ID {ARUCO_ID}: FOUND"
                )
            else:
                print(
                    f"ArUco ID {ARUCO_ID}: NOT FOUND"
                )

            print(
                f"Inference: {inference_ms:.1f} ms"
            )


            # =================================================
            # PROCESS OBJECTS
            # =================================================

            for object_number, box in enumerate(
                result.boxes,
                start=1
            ):

                class_id = int(
                    box.cls[0].item()
                )

                confidence = float(
                    box.conf[0].item()
                )

                shape = CLASS_NAMES.get(
                    class_id,
                    f"class_{class_id}"
                )

                xyxy = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )


                # ---------------------------------------------
                # CONTOUR
                # ---------------------------------------------

                contour = find_object_contour(
                    frame,
                    xyxy
                )

                if contour is None:
                    continue


                # ---------------------------------------------
                # CENTROID
                # ---------------------------------------------

                center = contour_centroid(
                    contour
                )

                if center is None:
                    continue


                cx, cy = center


                # ---------------------------------------------
                # COLOUR
                # ---------------------------------------------

                color, color_score = detect_color(
                    roi,
                    contour
                )


                # ---------------------------------------------
                # ORIENTATION
                # ---------------------------------------------

                angle = None

                arrow_tip = None

                if shape == "triangle":

                    orientation = (
                        triangle_orientation(
                            contour,
                            center
                        )
                    )

                    if orientation is not None:

                        angle, arrow_tip = orientation


                elif shape == "square":

                    angle = square_orientation(
                        contour
                    )


                # ---------------------------------------------
                # PIXEL -> MILLIMETRES
                # ---------------------------------------------

                marker_position = (
                    calibration.pixel_to_mm(
                        cx,
                        cy
                    )
                )


                # ---------------------------------------------
                # DRAW CONTOUR
                # ---------------------------------------------

                cv2.drawContours(
                    frame,
                    [contour],
                    -1,
                    (0, 255, 0),
                    2
                )

                cv2.circle(
                    frame,
                    (int(cx), int(cy)),
                    7,
                    (0, 0, 255),
                    -1
                )


                # Triangle orientation arrow

                if arrow_tip is not None:

                    cv2.arrowedLine(
                        frame,
                        (int(cx), int(cy)),
                        arrow_tip,
                        (255, 255, 0),
                        3,
                        tipLength=0.2
                    )


                # ---------------------------------------------
                # LABEL
                # ---------------------------------------------

                label = (
                    f"{color} {shape} "
                    f"{confidence:.2f}"
                )

                x1, y1, x2, y2 = map(
                    int,
                    xyxy
                )

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 255),
                    2
                )

                cv2.putText(
                    frame,
                    label,
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 255),
                    2
                )


                # ---------------------------------------------
                # TERMINAL OUTPUT
                # ---------------------------------------------

                print()
                print(
                    f"OBJECT {object_number}"
                )

                print(
                    f"  Shape       : {shape}"
                )

                print(
                    f"  Confidence  : {confidence:.3f}"
                )

                print(
                    f"  Color       : {color}"
                    f" ({color_score:.2f})"
                )

                print(
                    f"  Pixel       : "
                    f"({cx:.1f}, {cy:.1f})"
                )


                if marker_position is not None:

                    mx, my = marker_position

                    print(
                        f"  Marker XY   : "
                        f"({mx:.2f}, {my:.2f}) mm"
                    )

                else:

                    print(
                        "  Marker XY   : unavailable"
                    )


                if angle is None:

                    print(
                        "  Orientation : N/A"
                    )

                else:

                    print(
                        f"  Orientation : "
                        f"{angle:.1f} deg"
                    )


            # =================================================
            # DISPLAY STATUS
            # =================================================

            status = (
                f"ArUco {ARUCO_ID}: "
                + (
                    "CALIBRATED"
                    if calibration.H is not None
                    else "NOT FOUND"
                )
            )

            cv2.putText(
                frame,
                status,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Inference: {inference_ms:.1f} ms",
                (30, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            # Resize only for display.
            # Calculations still use full-resolution image.

            display = frame

            max_width = 1400

            if frame.shape[1] > max_width:

                scale = (
                    max_width
                    / frame.shape[1]
                )

                display = cv2.resize(
                    frame,
                    None,
                    fx=scale,
                    fy=scale
                )


            cv2.imshow(
                "INA700 - Basler Vision",
                display
            )


            key = (
                cv2.waitKey(1)
                & 0xFF
            )

            if key == ord("q"):
                break


    finally:

        camera.close()

        cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()