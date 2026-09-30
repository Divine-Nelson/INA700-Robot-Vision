from pypylon import pylon
import cv2
import os
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

SAVE_DIR = Path("fresh_test/abnomal")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_NAME = "Basler - Combine Dataset"


# ============================================================
# FIND NEXT IMAGE NUMBER
# ============================================================

existing = list(SAVE_DIR.glob("combine_*.jpg"))

numbers = []

for path in existing:
    try:
        number = int(path.stem.split("_")[1])
        numbers.append(number)
    except (ValueError, IndexError):
        pass

image_number = max(numbers, default=0) + 1


# ============================================================
# CONNECT TO BASLER CAMERA
# ============================================================

camera = pylon.InstantCamera(
    pylon.TlFactory.GetInstance().CreateFirstDevice()
)

camera.Open()

print("=" * 60)
print("Basler Combine Dataset Capture")
print("=" * 60)
print(f"Camera: {camera.GetDeviceInfo().GetModelName()}")
print(f"Saving to: {SAVE_DIR.resolve()}")
print()
print("SPACE = save image")
print("Q     = quit")
print("=" * 60)


# ============================================================
# IMAGE CONVERTER
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

saved_count = 0

try:

    while camera.IsGrabbing():

        grab_result = camera.RetrieveResult(
            5000,
            pylon.TimeoutHandling_ThrowException
        )

        if grab_result.GrabSucceeded():

            converted = converter.Convert(grab_result)

            frame = converted.GetArray()

            # ------------------------------------------------
            # Preview
            # ------------------------------------------------

            preview = frame.copy()

            cv2.putText(
                preview,
                f"Saved: {saved_count}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                preview,
                "SPACE = capture | Q = quit",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKey(1) & 0xFF

            # ------------------------------------------------
            # SAVE
            # ------------------------------------------------

            if key == ord(" "):

                filename = (
                    SAVE_DIR /
                    f"combine_{image_number:04d}.jpg"
                )

                success = cv2.imwrite(
                    str(filename),
                    frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 95]
                )

                if success:

                    print(f"[SAVED] {filename}")

                    image_number += 1
                    saved_count += 1

                else:
                    print("[ERROR] Could not save image")

            # ------------------------------------------------
            # QUIT
            # ------------------------------------------------

            elif key == ord("q"):
                break

        grab_result.Release()


finally:

    if camera.IsGrabbing():
        camera.StopGrabbing()

    camera.Close()

    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print(f"Capture finished.")
    print(f"Images saved this session: {saved_count}")
    print(f"Dataset location: {SAVE_DIR.resolve()}")
    print("=" * 60)