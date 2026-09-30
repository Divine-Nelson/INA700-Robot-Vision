import time
import cv2
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

WINDOW_NAME = "INA700 - Basler Live YOLO V3"


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("INA700 LIVE BASLER + YOLO V3")
print("=" * 70)

print("\nLoading model...")

model = YOLO(MODEL_PATH)

print("Model loaded.")
print(f"Classes: {model.names}")


# ============================================================
# CONNECT BASLER CAMERA
# ============================================================

print("\nConnecting to Basler camera...")

camera = pylon.InstantCamera(
    pylon.TlFactory.GetInstance().CreateFirstDevice()
)

camera.Open()

print(f"Camera: {camera.GetDeviceInfo().GetModelName()}")


# ============================================================
# PIXEL FORMAT CONVERTER
# ============================================================

converter = pylon.ImageFormatConverter()

converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


# ============================================================
# START CAMERA
# ============================================================

camera.StartGrabbing(
    pylon.GrabStrategy_LatestImageOnly
)

print("\nLive detection started.")
print("Press Q to quit.")
print("=" * 70)


# ============================================================
# FPS TRACKING
# ============================================================

previous_time = time.perf_counter()

fps = 0.0


# ============================================================
# MAIN LOOP
# ============================================================

try:

    while camera.IsGrabbing():

        grab_result = camera.RetrieveResult(
            5000,
            pylon.TimeoutHandling_ThrowException
        )

        if not grab_result.GrabSucceeded():

            print(
                "Grab failed:",
                grab_result.ErrorCode,
                grab_result.ErrorDescription
            )

            grab_result.Release()
            continue


        # ----------------------------------------------------
        # BASLER IMAGE -> OPENCV BGR IMAGE
        # ----------------------------------------------------

        converted = converter.Convert(grab_result)

        frame = converted.GetArray()

        grab_result.Release()


        # ----------------------------------------------------
        # YOLO INFERENCE
        # ----------------------------------------------------

        inference_start = time.perf_counter()

        results = model.predict(
            source=frame,
            imgsz=IMAGE_SIZE,
            conf=CONFIDENCE,
            device="cpu",
            verbose=False
        )

        inference_ms = (
            time.perf_counter() - inference_start
        ) * 1000


        # ----------------------------------------------------
        # DRAW YOLO RESULTS
        # ----------------------------------------------------

        result = results[0]

        annotated_frame = result.plot()


        # ----------------------------------------------------
        # CALCULATE FPS
        # ----------------------------------------------------

        current_time = time.perf_counter()

        elapsed = current_time - previous_time

        if elapsed > 0:
            fps = 1.0 / elapsed

        previous_time = current_time


        # ----------------------------------------------------
        # DISPLAY PERFORMANCE
        # ----------------------------------------------------

        cv2.putText(
            annotated_frame,
            f"Inference: {inference_ms:.1f} ms",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # PRINT DETECTIONS
        # ----------------------------------------------------

        if result.boxes is not None:

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                x1, y1, x2, y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )

                center_x = (
                    x1 + x2
                ) / 2

                center_y = (
                    y1 + y2
                ) / 2

                print(
                    f"{model.names[class_id]:8s} "
                    f"conf={confidence:.2f} "
                    f"center=({center_x:.1f}, {center_y:.1f})"
                )


        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        cv2.imshow(
            WINDOW_NAME,
            annotated_frame
        )


        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

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