import cv2


class WebcamCamera:

    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open webcam."
            )

    def get_frame(self):
        success, frame = self.cap.read()

        if not success:
            return None

        return frame

    def release(self):
        self.cap.release()


class BaslerCamera:

    def __init__(self):
        try:
            from pypylon import pylon
        except ImportError:
            raise RuntimeError(
                "pypylon is not installed. "
                "Run: pip install pypylon"
            )

        self.pylon = pylon

        self.camera = pylon.InstantCamera(
            pylon.TlFactory
            .GetInstance()
            .CreateFirstDevice()
        )

        self.camera.Open()

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
            self.pylon.TimeoutHandling_ThrowException
        )

        try:
            if not grab.GrabSucceeded():
                return None

            converted = self.converter.Convert(grab)

            return converted.GetArray()

        finally:
            grab.Release()

    def release(self):

        if self.camera.IsGrabbing():
            self.camera.StopGrabbing()

        if self.camera.IsOpen():
            self.camera.Close()