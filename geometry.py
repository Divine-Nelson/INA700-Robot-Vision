import cv2
import numpy as np
import math


class GeometryAnalyzer:

    @staticmethod
    def analyze(frame, detection):
        """
        Extract precise center and orientation from a YOLO detection.

        YOLO gives us the region of interest.
        OpenCV analyzes the actual object contour.
        """

        x1, y1, x2, y2 = detection["bbox"]

        # Prevent invalid array indexes.
        h, w = frame.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return None

        contour = GeometryAnalyzer._largest_contour(roi)

        if contour is None:
            return None

        # Convert contour coordinates from ROI coordinates
        # back to the original camera frame.
        contour_global = contour.copy()

        contour_global[:, 0, 0] += x1
        contour_global[:, 0, 1] += y1

        center = GeometryAnalyzer._calculate_centroid(
            contour_global
        )

        if center is None:
            return None

        shape = detection["class_name"]

        if shape == "triangle":
            angle = GeometryAnalyzer._triangle_orientation(
                contour_global,
                center
            )

        elif shape == "square":
            angle = GeometryAnalyzer._square_orientation(
                contour_global
            )

        elif shape == "circle":

            # A geometrically uniform circle has no observable
            # rotation around its center.
            angle = None

        else:
            angle = None

        return {
            "center": center,
            "angle": angle,
            "contour": contour_global
        }

    @staticmethod
    def _largest_contour(roi):
        """
        Segment the object and return the largest contour.

        This is intentionally simple initially.
        We will tune segmentation using the real dataset/camera.
        """

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # Otsu automatically selects a threshold.
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        contours, _ = cv2.findContours(
            binary,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        # Remove very small noise.
        contours = [
            c for c in contours
            if cv2.contourArea(c) > 100
        ]

        if not contours:
            return None

        return max(
            contours,
            key=cv2.contourArea
        )

    @staticmethod
    def _calculate_centroid(contour):
        """
        Calculate the centroid using image moments.
        """

        M = cv2.moments(contour)

        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return cx, cy

    @staticmethod
    def _triangle_orientation(contour, center):
        """
        Estimate triangle orientation.

        We approximate the contour to three vertices and
        use the vertex furthest from the centroid as the
        triangle's pointing direction.
        """

        perimeter = cv2.arcLength(
            contour,
            True
        )

        polygon = cv2.approxPolyDP(
            contour,
            0.04 * perimeter,
            True
        )

        if len(polygon) != 3:
            return None

        cx, cy = center

        vertices = [
            point[0]
            for point in polygon
        ]

        # Find the vertex furthest from the center.
        tip = max(
            vertices,
            key=lambda p:
                (p[0] - cx) ** 2 +
                (p[1] - cy) ** 2
        )

        dx = tip[0] - cx

        # Image Y coordinates increase downward.
        dy = cy - tip[1]

        angle = math.degrees(
            math.atan2(dy, dx)
        )

        # Convert to 0–360 degrees.
        angle = (angle + 360) % 360

        return angle

    @staticmethod
    def _square_orientation(contour):
        """
        Estimate square orientation using a minimum-area
        rotated rectangle.
        """

        rect = cv2.minAreaRect(contour)

        (_, _), (width, height), angle = rect

        if width < height:
            angle += 90

        # Square has 90-degree rotational symmetry.
        angle = angle % 90

        return angle