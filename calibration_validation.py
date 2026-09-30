# -*- coding: utf-8 -*-

"""
INA700 - Camera Calibration Validation
---------------------------------------

Purpose:
    Validate pixel -> physical millimetre conversion using:
        Basler camera
        ArUco DICT_5X5_100, ID 21
        100 mm x 100 mm physical marker
        YOLO V3 object detection
        OpenCV contour centroid
        Planar homography

Coordinate system:

    Marker corner 0 = (0,   0) mm
    Marker corner 1 = (100, 0) mm
    Marker corner 2 = (100, 100) mm
    Marker corner 3 = (0,   100) mm

Controls:
    Q / ESC  -> quit
    S        -> print current measurements to terminal
"""

import time
import cv2
import numpy as np

from ultralytics import YOLO
from pypylon import pylon


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = (
    r"H:\ABB_Vision_Project"
    r"\runs\detect\ina700_3class_basler_v3"
    r"\weights\best.pt"
)

ARUCO_ID = 21
MARKER_SIZE_MM = 100.0

YOLO_CONF = 0.25
YOLO_IMGSZ = 640


# ============================================================
# BASLER CAMERA
# ============================================================

class BaslerCamera:

    def __init__(self):

        self.camera = None
        self.converter = None

    def open(self):

        factory = pylon.TlFactory.GetInstance()

        devices = factory.EnumerateDevices()

        if len(devices) == 0:
            raise RuntimeError(
                "No Basler camera detected."
            )

        self.camera = pylon.InstantCamera(
            factory.CreateDevice(devices[0])
        )

        self.camera.Open()

        print(
            "Camera:",
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

    def get_frame(self):

        grab = self.camera.RetrieveResult(
            5000,
            pylon.TimeoutHandling_ThrowException
        )

        try:

            if not grab.GrabSucceeded():

                return None

            converted = self.converter.Convert(grab)

            frame = converted.GetArray()

            return frame.copy()

        finally:

            grab.Release()

    def close(self):

        if self.camera is not None:

            if self.camera.IsGrabbing():
                self.camera.StopGrabbing()

            if self.camera.IsOpen():
                self.camera.Close()


# ============================================================
# ARUCO CALIBRATION
# ============================================================

class ArucoCalibration:

    def __init__(self):

        dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_5X5_100
        )

        parameters = cv2.aruco.DetectorParameters()

        self.detector = cv2.aruco.ArucoDetector(
            dictionary,
            parameters
        )

        self.H = None
        self.corners = None

    def detect(self, frame):

        corners, ids, _ = self.detector.detectMarkers(
            frame
        )

        self.H = None
        self.corners = None

        if ids is None:
            return False

        ids = ids.flatten()

        for marker_corners, marker_id in zip(
            corners,
            ids
        ):

            if int(marker_id) != ARUCO_ID:
                continue

            pts = marker_corners.reshape(
                4,
                2
            ).astype(np.float32)

            self.corners = pts

            destination = np.array(
                [
                    [0.0, 0.0],
                    [MARKER_SIZE_MM, 0.0],
                    [MARKER_SIZE_MM, MARKER_SIZE_MM],
                    [0.0, MARKER_SIZE_MM],
                ],
                dtype=np.float32
            )

            self.H = cv2.getPerspectiveTransform(
                pts,
                destination
            )

            return True

        return False

    def pixel_to_mm(self, x, y):

        if self.H is None:
            return None

        point = np.array(
            [[[float(x), float(y)]]],
            dtype=np.float32
        )

        transformed = cv2.perspectiveTransform(
            point,
            self.H
        )

        X = float(transformed[0, 0, 0])
        Y = float(transformed[0, 0, 1])

        return X, Y

    def draw(self, frame):

        if self.corners is None:
            return

        pts = self.corners.astype(int)

        # Draw marker boundary
        for i in range(4):

            p1 = tuple(pts[i])

            p2 = tuple(
                pts[(i + 1) % 4]
            )

            cv2.line(
                frame,
                p1,
                p2,
                (255, 255, 255),
                2
            )

        # Draw corner numbers
        for i, point in enumerate(pts):

            cv2.circle(
                frame,
                tuple(point),
                6,
                (255, 255, 255),
                -1
            )

            cv2.putText(
                frame,
                str(i),
                (
                    int(point[0]) + 8,
                    int(point[1]) - 8
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

        # Marker centre
        centre = np.mean(
            self.corners,
            axis=0
        )

        cx = int(centre[0])
        cy = int(centre[1])

        cv2.circle(
            frame,
            (cx, cy),
            7,
            (255, 255, 255),
            -1
        )

        cv2.putText(
            frame,
            "Marker centre = (50,50) mm",
            (cx + 12, cy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )


# ============================================================
# CONTOUR EXTRACTION
# ============================================================

def extract_object_contour(roi):

    if roi is None or roi.size == 0:
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

    # Try normal threshold
    _, binary_normal = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY
        + cv2.THRESH_OTSU
    )

    # Try inverted threshold
    _, binary_inverse = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )

    roi_area = (
        roi.shape[0]
        * roi.shape[1]
    )

    candidates = []

    for binary in [
        binary_normal,
        binary_inverse
    ]:

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(
                contour
            )

            if area < 100:
                continue

            # Reject contours that are basically
            # the complete YOLO ROI.
            if area > 0.90 * roi_area:
                continue

            candidates.append(
                contour
            )

    if not candidates:
        return None

    return max(
        candidates,
        key=cv2.contourArea
    )


# ============================================================
# CENTROID
# ============================================================

def contour_centroid(contour):

    if contour is None:
        return None

    M = cv2.moments(
        contour
    )

    if abs(M["m00"]) < 1e-6:
        return None

    cx = (
        M["m10"]
        / M["m00"]
    )

    cy = (
        M["m01"]
        / M["m00"]
    )

    return float(cx), float(cy)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("INA700 CAMERA CALIBRATION VALIDATION")
    print("=" * 70)

    print()
    print("Loading YOLO V3...")

    model = YOLO(
        MODEL_PATH
    )

    print("Model loaded.")

    camera = BaslerCamera()

    calibration = ArucoCalibration()

    camera.open()

    print()
    print("ArUco:")
    print(
        f"  Dictionary : DICT_5X5_100"
    )
    print(
        f"  ID         : {ARUCO_ID}"
    )
    print(
        f"  Size       : {MARKER_SIZE_MM:.1f} mm"
    )

    print()
    print("Controls:")
    print("  S = print measurements")
    print("  Q = quit")
    print()

    last_measurements = []

    try:

        while True:

            frame = camera.get_frame()

            if frame is None:
                continue

            display = frame.copy()

            # ==================================================
            # ARUCO
            # ==================================================

            marker_found = calibration.detect(
                frame
            )

            if marker_found:

                calibration.draw(
                    display
                )

                cv2.putText(
                    display,
                    "ARUCO 21: FOUND",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

            else:

                cv2.putText(
                    display,
                    "ARUCO 21: NOT FOUND",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )


            # ==================================================
            # YOLO
            # ==================================================

            start = time.perf_counter()

            results = model.predict(
                frame,
                imgsz=YOLO_IMGSZ,
                conf=YOLO_CONF,
                device="cpu",
                verbose=False
            )

            inference_ms = (
                time.perf_counter()
                - start
            ) * 1000.0


            last_measurements = []


            # ==================================================
            # PROCESS DETECTIONS
            # ==================================================

            for result in results:

                if result.boxes is None:
                    continue

                for box in result.boxes:

                    class_id = int(
                        box.cls[0]
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    shape = model.names[
                        class_id
                    ]


                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                        .astype(int)
                    )


                    # Clamp to image
                    x1 = max(
                        0,
                        min(
                            x1,
                            frame.shape[1] - 1
                        )
                    )

                    y1 = max(
                        0,
                        min(
                            y1,
                            frame.shape[0] - 1
                        )
                    )

                    x2 = max(
                        x1 + 1,
                        min(
                            x2,
                            frame.shape[1]
                        )
                    )

                    y2 = max(
                        y1 + 1,
                        min(
                            y2,
                            frame.shape[0]
                        )
                    )


                    roi = frame[
                        y1:y2,
                        x1:x2
                    ]


                    # ------------------------------------------
                    # CONTOUR
                    # ------------------------------------------

                    contour = (
                        extract_object_contour(
                            roi
                        )
                    )

                    centroid = (
                        contour_centroid(
                            contour
                        )
                    )

                    if centroid is None:
                        continue


                    roi_cx, roi_cy = (
                        centroid
                    )


                    # Convert ROI coordinates back to
                    # full-frame pixel coordinates.
                    pixel_x = (
                        x1
                        + roi_cx
                    )

                    pixel_y = (
                        y1
                        + roi_cy
                    )


                    # ------------------------------------------
                    # DRAW CONTOUR
                    # ------------------------------------------

                    contour_global = (
                        contour.copy()
                    )

                    contour_global[
                        :,
                        0,
                        0
                    ] += x1

                    contour_global[
                        :,
                        0,
                        1
                    ] += y1


                    cv2.drawContours(
                        display,
                        [contour_global],
                        -1,
                        (0, 255, 0),
                        2
                    )


                    # YOLO bounding box
                    cv2.rectangle(
                        display,
                        (x1, y1),
                        (x2, y2),
                        (255, 0, 255),
                        2
                    )


                    # Centroid
                    cv2.circle(
                        display,
                        (
                            int(pixel_x),
                            int(pixel_y)
                        ),
                        7,
                        (0, 0, 255),
                        -1
                    )


                    # ------------------------------------------
                    # PIXEL -> MILLIMETRES
                    # ------------------------------------------

                    marker_xy = None

                    if marker_found:

                        marker_xy = (
                            calibration.pixel_to_mm(
                                pixel_x,
                                pixel_y
                            )
                        )


                    # ------------------------------------------
                    # LABEL
                    # ------------------------------------------

                    if marker_xy is not None:

                        X_mm, Y_mm = marker_xy

                        label = (
                            f"{shape} "
                            f"({X_mm:.1f}, "
                            f"{Y_mm:.1f}) mm"
                        )

                    else:

                        X_mm = None
                        Y_mm = None

                        label = (
                            f"{shape} "
                            f"pixel "
                            f"({pixel_x:.0f},"
                            f"{pixel_y:.0f})"
                        )


                    cv2.putText(
                        display,
                        label,
                        (
                            x1,
                            max(
                                25,
                                y1 - 10
                            )
                        ),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 0, 255),
                        2
                    )


                    last_measurements.append(
                        {
                            "shape": shape,
                            "confidence": confidence,
                            "pixel_x": pixel_x,
                            "pixel_y": pixel_y,
                            "X_mm": X_mm,
                            "Y_mm": Y_mm,
                        }
                    )


            # ==================================================
            # STATUS
            # ==================================================

            cv2.putText(
                display,
                f"Inference: {inference_ms:.1f} ms",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )


            cv2.imshow(
                "INA700 Calibration Validation",
                display
            )


            key = (
                cv2.waitKey(1)
                & 0xFF
            )


            # Q or ESC
            if key in [
                ord("q"),
                ord("Q"),
                27
            ]:
                break


            # ==================================================
            # SAVE / PRINT CURRENT MEASUREMENT
            # ==================================================

            if key in [
                ord("s"),
                ord("S")
            ]:

                print()
                print("=" * 70)
                print(
                    "CURRENT CALIBRATION MEASUREMENT"
                )
                print("=" * 70)

                if not marker_found:

                    print(
                        "ArUco ID 21 not detected."
                    )

                elif not last_measurements:

                    print(
                        "No valid object detected."
                    )

                else:

                    for i, measurement in enumerate(
                        last_measurements,
                        start=1
                    ):

                        print()
                        print(
                            f"OBJECT {i}"
                        )

                        print(
                            f"  Shape      : "
                            f"{measurement['shape']}"
                        )

                        print(
                            f"  Confidence : "
                            f"{measurement['confidence']:.3f}"
                        )

                        print(
                            f"  Pixel      : "
                            f"("
                            f"{measurement['pixel_x']:.2f}, "
                            f"{measurement['pixel_y']:.2f}"
                            f")"
                        )

                        if (
                            measurement["X_mm"]
                            is not None
                        ):

                            print(
                                f"  Marker XY  : "
                                f"("
                                f"{measurement['X_mm']:.2f}, "
                                f"{measurement['Y_mm']:.2f}"
                                f") mm"
                            )

                print("=" * 70)


    finally:

        camera.close()

        cv2.destroyAllWindows()

        print()
        print(
            "Calibration validation stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()