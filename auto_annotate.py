"""
INA700 - One-Click YOLO Annotator
=================================

Classes:
    0 = square
    1 = circle

Workflow:
    1. Image appears.
    2. Click ONCE inside the actual target object.
    3. OpenCV finds the coloured region connected to your click.
    4. A proposed bounding box is shown.
    5. Press:
         Y = accept
         R = redraw manually
         C = click again
         S = skip
         D = reject image
         Q = quit

Already-labelled images are automatically skipped.

IMPORTANT:
Only the actual assigned square/circle should be annotated.
Small circles, stars, ArUco markers and other objects are distractors.
"""

from pathlib import Path
import shutil

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path("dataset")

SPLITS = ["train", "val"]

CLASS_IDS = {
    "square": 0,
    "circle": 1,
}

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# GLOBAL CLICK POSITION
# ============================================================

clicked_point = None


def mouse_callback(event, x, y, flags, param):
    """
    Store the location clicked by the user.
    """

    global clicked_point

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)


# ============================================================
# CLASS FROM FILENAME
# ============================================================

def get_class_from_filename(path):

    name = path.stem.lower()

    if name.startswith("square_"):
        return "square"

    if name.startswith("circle_"):
        return "circle"

    return None


# ============================================================
# FIND OBJECT FROM CLICK
# ============================================================

def find_object_from_click(image, point):
    """
    Find the coloured object surrounding the clicked point.

    Instead of trying to decide which object is the target,
    the human tells OpenCV by clicking inside it.

    OpenCV then performs HSV colour segmentation around the
    colour of the clicked pixel.
    """

    if point is None:
        return None, None

    click_x, click_y = point

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # --------------------------------------------------------
    # Use a small region around the click instead of ONE pixel.
    #
    # This makes the colour estimate more robust against
    # reflections, noise and shadows.
    # --------------------------------------------------------

    radius = 5

    y1 = max(
        click_y - radius,
        0
    )

    y2 = min(
        click_y + radius + 1,
        hsv.shape[0]
    )

    x1 = max(
        click_x - radius,
        0
    )

    x2 = min(
        click_x + radius + 1,
        hsv.shape[1]
    )

    sample = hsv[
        y1:y2,
        x1:x2
    ]

    # Median colour is more robust than using one pixel.
    h = int(
        np.median(
            sample[:, :, 0]
        )
    )

    s = int(
        np.median(
            sample[:, :, 1]
        )
    )

    v = int(
        np.median(
            sample[:, :, 2]
        )
    )

    print(
        f"Clicked HSV: "
        f"H={h}, S={s}, V={v}"
    )

    # --------------------------------------------------------
    # HSV tolerance
    # --------------------------------------------------------

    H_TOLERANCE = 15
    S_TOLERANCE = 90
    V_TOLERANCE = 100

    lower_h = max(
        h - H_TOLERANCE,
        0
    )

    upper_h = min(
        h + H_TOLERANCE,
        179
    )

    lower_s = max(
        s - S_TOLERANCE,
        20
    )

    upper_s = min(
        s + S_TOLERANCE,
        255
    )

    # Allow very dark coloured targets.
    # Some square objects have V values below 25.
    lower_v = max(
        v - V_TOLERANCE,
        5
    )

    upper_v = min(
        v + V_TOLERANCE,
        255
    )

    lower = np.array([
        lower_h,
        lower_s,
        lower_v
    ])

    upper = np.array([
        upper_h,
        upper_s,
        upper_v
    ])

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    # --------------------------------------------------------
    # Clean segmentation
    # --------------------------------------------------------

    kernel = np.ones(
        (5, 5),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # --------------------------------------------------------
    # Find connected components
    #
    # IMPORTANT:
    # We don't simply choose the largest coloured object.
    #
    # We choose the component containing the user's click.
    # --------------------------------------------------------

    count, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    component_id = labels[
        click_y,
        click_x
    ]

    if component_id == 0:
        print(
            "Click did not fall inside "
            "a detected coloured region."
        )

        return None, mask

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    x = stats[
        component_id,
        cv2.CC_STAT_LEFT
    ]

    y = stats[
        component_id,
        cv2.CC_STAT_TOP
    ]

    w = stats[
        component_id,
        cv2.CC_STAT_WIDTH
    ]

    h_box = stats[
        component_id,
        cv2.CC_STAT_HEIGHT
    ]

    area = stats[
        component_id,
        cv2.CC_STAT_AREA
    ]

    print(
        f"Detected component: "
        f"x={x}, y={y}, "
        f"w={w}, h={h_box}, "
        f"area={area}"
    )

    bbox = (
        int(x),
        int(y),
        int(w),
        int(h_box)
    )

    return bbox, mask


# ============================================================
# YOLO CONVERSION
# ============================================================

def bbox_to_yolo(
    bbox,
    image_width,
    image_height
):

    x, y, w, h = bbox

    center_x = (
        x + w / 2
    ) / image_width

    center_y = (
        y + h / 2
    ) / image_height

    width = (
        w / image_width
    )

    height = (
        h / image_height
    )

    return (
        center_x,
        center_y,
        width,
        height
    )


# ============================================================
# SAVE YOLO LABEL
# ============================================================

def save_label(
    label_path,
    class_id,
    bbox,
    image_shape
):

    image_height, image_width = (
        image_shape[:2]
    )

    (
        center_x,
        center_y,
        width,
        height
    ) = bbox_to_yolo(
        bbox,
        image_width,
        image_height
    )

    label_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    text = (
        f"{class_id} "
        f"{center_x:.6f} "
        f"{center_y:.6f} "
        f"{width:.6f} "
        f"{height:.6f}\n"
    )

    label_path.write_text(
        text,
        encoding="utf-8"
    )


# ============================================================
# MANUAL BOUNDING BOX
# ============================================================

def manual_bbox(image):

    print()
    print(
        "Draw a box around the target."
    )

    print(
        "Press ENTER or SPACE when finished."
    )

    bbox = cv2.selectROI(
        "Manual Bounding Box",
        image,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyWindow(
        "Manual Bounding Box"
    )

    x, y, w, h = map(
        int,
        bbox
    )

    if w == 0 or h == 0:
        return None

    return (
        x,
        y,
        w,
        h
    )


# ============================================================
# REJECT
# ============================================================

def reject_image(
    image_path,
    split
):

    rejected_folder = (
        DATASET /
        "rejected" /
        split
    )

    rejected_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    destination = (
        rejected_folder /
        image_path.name
    )

    shutil.move(
        str(image_path),
        str(destination)
    )

    print(
        f"[REJECTED] {image_path.name}"
    )


# ============================================================
# DRAW DISPLAY
# ============================================================

def create_display(
    image,
    bbox,
    target_class,
    index,
    total,
    filename,
    point=None,
):

    display = image.copy()

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    if bbox is not None:

        x, y, w, h = bbox

        cv2.rectangle(
            display,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        cv2.putText(
            display,
            target_class.upper(),
            (
                x,
                max(y - 10, 30)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    # --------------------------------------------------------
    # Click point
    # --------------------------------------------------------

    if point is not None:

        cv2.drawMarker(
            display,
            point,
            (0, 255, 255),
            cv2.MARKER_CROSS,
            25,
            3
        )

    # --------------------------------------------------------
    # Information panel
    # --------------------------------------------------------

    cv2.rectangle(
        display,
        (0, 0),
        (
            display.shape[1],
            110
        ),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        display,
        f"{index}/{total}   {filename}",
        (15, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        display,
        f"Target: {target_class.upper()}",
        (15, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    if bbox is None:

        instruction = (
            "CLICK inside the target object"
        )

    else:

        instruction = (
            "[Y] Accept  "
            "[C] Click again  "
            "[R] Manual  "
            "[S] Skip  "
            "[D] Reject  "
            "[Q] Quit"
        )

    cv2.putText(
        display,
        instruction,
        (15, 83),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        2
    )

    return display


# ============================================================
# PROCESS SPLIT
# ============================================================

def process_split(split):

    global clicked_point

    image_folder = (
        DATASET /
        "images" /
        split
    )

    label_folder = (
        DATASET /
        "labels" /
        split
    )

    label_folder.mkdir(
        parents=True,
        exist_ok=True
    )

    images = sorted([
        path
        for path in image_folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ])

    print()
    print("=" * 60)
    print(
        f"PROCESSING {split.upper()}"
    )
    print("=" * 60)

    for index, image_path in enumerate(
        images,
        start=1
    ):

        target_class = (
            get_class_from_filename(
                image_path
            )
        )

        if target_class is None:

            print(
                f"Unknown class: "
                f"{image_path.name}"
            )

            continue

        label_path = (
            label_folder /
            f"{image_path.stem}.txt"
        )

        # Resume automatically
        if label_path.exists():
            continue

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"Cannot read: "
                f"{image_path.name}"
            )

            continue

        bbox = None
        clicked_point = None

        # ----------------------------------------------------
        # Main annotation loop
        # ----------------------------------------------------

        while True:

            display = create_display(
                image,
                bbox,
                target_class,
                index,
                len(images),
                image_path.name,
                clicked_point
            )

            # ------------------------------------------------
            # Resize display if necessary
            # ------------------------------------------------

            h, w = display.shape[:2]

            scale = min(
                1.0,
                1400 / w,
                850 / h
            )

            if scale < 1:

                preview = cv2.resize(
                    display,
                    None,
                    fx=scale,
                    fy=scale,
                    interpolation=cv2.INTER_AREA
                )

            else:

                preview = display

            window_name = (
                "INA700 One-Click Annotator"
            )

            cv2.imshow(
                window_name,
                preview
            )

            # ------------------------------------------------
            # IMPORTANT:
            # Mouse coordinates are relative to preview.
            #
            # Convert them back to original-image coordinates.
            # ------------------------------------------------

            temporary_click = [None]

            def scaled_mouse_callback(
                event,
                x,
                y,
                flags,
                param
            ):

                if event == cv2.EVENT_LBUTTONDOWN:

                    original_x = int(
                        x / scale
                    )

                    original_y = int(
                        y / scale
                    )

                    temporary_click[0] = (
                        original_x,
                        original_y
                    )

            cv2.setMouseCallback(
                window_name,
                scaled_mouse_callback
            )

            key = (
                cv2.waitKey(0)
                & 0xFF
            )

            # ------------------------------------------------
            # Mouse click
            # ------------------------------------------------

            if temporary_click[0] is not None:

                clicked_point = (
                    temporary_click[0]
                )

                bbox, mask = (
                    find_object_from_click(
                        image,
                        clicked_point
                    )
                )

                continue

            # ------------------------------------------------
            # ACCEPT
            # ------------------------------------------------

            if key in (
                ord("y"),
                ord("Y")
            ):

                if bbox is None:

                    print(
                        "Click the target first."
                    )

                    continue

                save_label(
                    label_path,
                    CLASS_IDS[
                        target_class
                    ],
                    bbox,
                    image.shape
                )

                print(
                    f"[SAVED] "
                    f"{label_path.name}"
                )

                break

            # ------------------------------------------------
            # CLICK AGAIN
            # ------------------------------------------------

            elif key in (
                ord("c"),
                ord("C")
            ):

                bbox = None
                clicked_point = None

                continue

            # ------------------------------------------------
            # MANUAL
            # ------------------------------------------------

            elif key in (
                ord("r"),
                ord("R")
            ):

                new_bbox = manual_bbox(
                    image
                )

                if new_bbox is not None:
                    bbox = new_bbox

                continue

            # ------------------------------------------------
            # SKIP
            # ------------------------------------------------

            elif key in (
                ord("s"),
                ord("S")
            ):

                print(
                    f"[SKIPPED] "
                    f"{image_path.name}"
                )

                break

            # ------------------------------------------------
            # REJECT
            # ------------------------------------------------

            elif key in (
                ord("d"),
                ord("D")
            ):

                reject_image(
                    image_path,
                    split
                )

                break

            # ------------------------------------------------
            # QUIT
            # ------------------------------------------------

            elif key in (
                ord("q"),
                ord("Q")
            ):

                cv2.destroyAllWindows()

                print()
                print(
                    "Stopped safely."
                )

                return False

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "INA700 ONE-CLICK YOLO ANNOTATOR"
    )
    print("=" * 60)

    print()
    print("Classes:")
    print("  0 = square")
    print("  1 = circle")

    print()
    print(
        "Click inside the actual target object."
    )

    print()
    print("Controls:")
    print("  Mouse = select target")
    print("  Y     = accept")
    print("  C     = click again")
    print("  R     = manual box")
    print("  S     = skip")
    print("  D     = reject")
    print("  Q     = quit")

    for split in SPLITS:

        success = process_split(
            split
        )

        if not success:
            return

    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("ANNOTATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()