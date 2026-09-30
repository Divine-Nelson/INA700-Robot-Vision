import cv2
from pypylon import pylon


# ============================================================
# ARUCO CONFIGURATION
# ============================================================

ARUCO_DICTIONARY = cv2.aruco.DICT_5X5_100
TARGET_MARKER_ID = 21

dictionary = cv2.aruco.getPredefinedDictionary(
    ARUCO_DICTIONARY
)

parameters = cv2.aruco.DetectorParameters()

detector = cv2.aruco.ArucoDetector(
    dictionary,
    parameters
)


# ============================================================
# BASLER CAMERA
# ============================================================

camera = pylon.InstantCamera(
    pylon.TlFactory.GetInstance().CreateFirstDevice()
)

camera.Open()

print("=" * 70)
print("INA700 - LIVE ARUCO TEST")
print("=" * 70)
print("Camera:", camera.GetDeviceInfo().GetModelName())
print("Dictionary: DICT_5X5_100")
print("Looking for marker ID 0")
print("Press Q to quit.")
print("=" * 70)


converter = pylon.ImageFormatConverter()

converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

camera.StartGrabbing(
    pylon.GrabStrategy_LatestImageOnly
)


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

        converted = converter.Convert(grab)
        frame = converted.GetArray()

        grab.Release()

        display = frame.copy()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        corners, ids, rejected = detector.detectMarkers(gray)

        # ----------------------------------------------------
        # MARKER FOUND
        # ----------------------------------------------------

        if ids is not None:

            cv2.aruco.drawDetectedMarkers(
                display,
                corners,
                ids
            )

            for marker_corners, marker_id in zip(
                corners,
                ids.flatten()
            ):

                if marker_id != TARGET_MARKER_ID:
                    continue

                pts = marker_corners.reshape(4, 2)

                # OpenCV ArUco corner order:
                #
                # 0 = top-left
                # 1 = top-right
                # 2 = bottom-right
                # 3 = bottom-left

                top_left = pts[0]
                top_right = pts[1]
                bottom_right = pts[2]
                bottom_left = pts[3]

                center_x = pts[:, 0].mean()
                center_y = pts[:, 1].mean()

                cv2.circle(
                    display,
                    (int(center_x), int(center_y)),
                    8,
                    (255, 255, 255),
                    -1
                )

                cv2.putText(
                    display,
                    "ID 21",
                    (int(center_x) + 15, int(center_y)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2
                )

                print(
                    "\r"
                    f"ID 21 | "
                    f"TL=({top_left[0]:.1f},{top_left[1]:.1f}) "
                    f"TR=({top_right[0]:.1f},{top_right[1]:.1f}) "
                    f"BR=({bottom_right[0]:.1f},{bottom_right[1]:.1f}) "
                    f"BL=({bottom_left[0]:.1f},{bottom_left[1]:.1f}) "
                    f"C=({center_x:.1f},{center_y:.1f})",
                    end=""
                )

        else:

            cv2.putText(
                display,
                "ARUCO ID 21 NOT DETECTED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2
            )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        cv2.namedWindow(
            "INA700 - ArUco Calibration Test",
            cv2.WINDOW_NORMAL
        )

        cv2.imshow(
            "INA700 - ArUco Calibration Test",
            display
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

finally:

    print("\n\nStopping camera...")

    if camera.IsGrabbing():
        camera.StopGrabbing()

    camera.Close()

    cv2.destroyAllWindows()

    print("Camera closed.")