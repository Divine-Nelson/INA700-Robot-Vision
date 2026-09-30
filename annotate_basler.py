import cv2
import numpy as np
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOT = Path("basler_raw")

CLASS_INFO = {
    "square_raw": 0,
    "circle_raw": 1,
    "triangle_raw": 2,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# HSV tolerances around clicked pixel
H_TOLERANCE = 15
S_TOLERANCE = 90
V_TOLERANCE = 100

# Preview size only — original image is NOT resized
MAX_PREVIEW_WIDTH = 1200
MAX_PREVIEW_HEIGHT = 800


# ============================================================
# GLOBAL MOUSE STATE
# ============================================================

clicked_point = None


def mouse_callback(event, x, y, flags, param):
    global clicked_point

    if event == cv2.EVENT_LBUTTONDOWN:
        scale = param

        original_x = int(x / scale)
        original_y = int(y / scale)

        clicked_point = (original_x, original_y)


# ============================================================
# IMAGE UTILITIES
# ============================================================

def get_preview_scale(image):
    h, w = image.shape[:2]

    scale_w = MAX_PREVIEW_WIDTH / w
    scale_h = MAX_PREVIEW_HEIGHT / h

    return min(1.0, scale_w, scale_h)


def make_preview(image, scale):
    if scale == 1.0:
        return image.copy()

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


# ============================================================
# HSV TARGET EXTRACTION
# ============================================================

def find_target_from_click(image, point):
    x, y = point
    h_img, w_img = image.shape[:2]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Sample small region around click
    radius = 5

    x1 = max(0, x - radius)
    x2 = min(w_img, x + radius + 1)

    y1 = max(0, y - radius)
    y2 = min(h_img, y + radius + 1)

    patch = hsv[y1:y2, x1:x2]

    if patch.size == 0:
        return None, None

    median = np.median(
        patch.reshape(-1, 3),
        axis=0,
    )

    h, s, v = median

    lower_s = max(int(s) - S_TOLERANCE, 20)
    upper_s = min(int(s) + S_TOLERANCE, 255)

    lower_v = max(int(v) - V_TOLERANCE, 5)
    upper_v = min(int(v) + V_TOLERANCE, 255)

    # --------------------------------------------------------
    # Hue wraparound handling
    # --------------------------------------------------------

    lower_h = int(h) - H_TOLERANCE
    upper_h = int(h) + H_TOLERANCE

    if lower_h < 0:
        mask1 = cv2.inRange(
            hsv,
            np.array([0, lower_s, lower_v]),
            np.array([upper_h, upper_s, upper_v]),
        )

        mask2 = cv2.inRange(
            hsv,
            np.array([180 + lower_h, lower_s, lower_v]),
            np.array([179, upper_s, upper_v]),
        )

        mask = cv2.bitwise_or(mask1, mask2)

    elif upper_h > 179:
        mask1 = cv2.inRange(
            hsv,
            np.array([lower_h, lower_s, lower_v]),
            np.array([179, upper_s, upper_v]),
        )

        mask2 = cv2.inRange(
            hsv,
            np.array([0, lower_s, lower_v]),
            np.array([upper_h - 180, upper_s, upper_v]),
        )

        mask = cv2.bitwise_or(mask1, mask2)

    else:
        mask = cv2.inRange(
            hsv,
            np.array([lower_h, lower_s, lower_v]),
            np.array([upper_h, upper_s, upper_v]),
        )

    # --------------------------------------------------------
    # Clean mask
    # --------------------------------------------------------

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )

    selected_label = None

    # Prefer component containing clicked point
    if 0 <= y < labels.shape[0] and 0 <= x < labels.shape[1]:
        label_at_click = labels[y, x]

        if label_at_click != 0:
            selected_label = label_at_click

    # If exact clicked pixel wasn't included,
    # inspect a small neighbourhood
    if selected_label is None:
        search_radius = 10

        sx1 = max(0, x - search_radius)
        sx2 = min(w_img, x + search_radius + 1)

        sy1 = max(0, y - search_radius)
        sy2 = min(h_img, y + search_radius + 1)

        neighbourhood = labels[sy1:sy2, sx1:sx2]

        candidates = neighbourhood[
            neighbourhood != 0
        ]

        if candidates.size > 0:
            values, counts = np.unique(
                candidates,
                return_counts=True,
            )

            selected_label = values[
                np.argmax(counts)
            ]

    if selected_label is None:
        return None, mask

    x_box = stats[selected_label, cv2.CC_STAT_LEFT]
    y_box = stats[selected_label, cv2.CC_STAT_TOP]
    w_box = stats[selected_label, cv2.CC_STAT_WIDTH]
    h_box = stats[selected_label, cv2.CC_STAT_HEIGHT]
    area = stats[selected_label, cv2.CC_STAT_AREA]

    if area < 100:
        return None, mask

    bbox = (
        int(x_box),
        int(y_box),
        int(w_box),
        int(h_box),
    )

    return bbox, mask


# ============================================================
# YOLO LABEL
# ============================================================

def bbox_to_yolo(bbox, image_width, image_height):
    x, y, w, h = bbox

    cx = (x + w / 2.0) / image_width
    cy = (y + h / 2.0) / image_height

    nw = w / image_width
    nh = h / image_height

    return cx, cy, nw, nh


def save_yolo_label(label_path, class_id, bbox, image_shape):
    h_img, w_img = image_shape[:2]

    cx, cy, w, h = bbox_to_yolo(
        bbox,
        w_img,
        h_img,
    )

    with open(label_path, "w") as f:
        f.write(
            f"{class_id} "
            f"{cx:.6f} "
            f"{cy:.6f} "
            f"{w:.6f} "
            f"{h:.6f}\n"
        )


# ============================================================
# DRAW
# ============================================================

def draw_bbox(image, bbox, class_name):
    x, y, w, h = bbox

    output = image.copy()

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3,
    )

    cv2.putText(
        output,
        class_name.upper(),
        (x, max(30, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )

    return output


# ============================================================
# ANNOTATION
# ============================================================

def annotate_image(image_path, class_name, class_id):
    global clicked_point

    label_path = image_path.with_suffix(".txt")

    # Resume-safe
    if label_path.exists():
        return "existing"

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        return "error"

    scale = get_preview_scale(image)

    window_name = "Basler YOLO Annotator"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL,
    )

    cv2.setMouseCallback(
        window_name,
        mouse_callback,
        scale,
    )

    bbox = None
    clicked_point = None

    while True:
        display = image.copy()

        if bbox is not None:
            display = draw_bbox(
                display,
                bbox,
                class_name,
            )

        preview = make_preview(
            display,
            scale,
        )

        instructions = (
            f"{class_name.upper()} | "
            "CLICK target | "
            "Y=accept | "
            "C=click again | "
            "R=manual | "
            "S=skip | "
            "Q=quit"
        )

        cv2.putText(
            preview,
            instructions,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
        )

        cv2.imshow(
            window_name,
            preview,
        )

        key = cv2.waitKey(30) & 0xFF

        # ----------------------------------------------------
        # Process new mouse click
        # ----------------------------------------------------

        if clicked_point is not None:
            bbox, mask = find_target_from_click(
                image,
                clicked_point,
            )

            clicked_point = None

        # Y = accept
        if key in (ord("y"), ord("Y")):
            if bbox is not None:
                save_yolo_label(
                    label_path,
                    class_id,
                    bbox,
                    image.shape,
                )

                return "accepted"

        # C = clear/reclick
        elif key in (ord("c"), ord("C")):
            bbox = None
            clicked_point = None

        # R = manual rectangle
        elif key in (ord("r"), ord("R")):
            manual_preview = make_preview(
                image,
                scale,
            )

            roi = cv2.selectROI(
                "Manual Bounding Box",
                manual_preview,
                showCrosshair=True,
                fromCenter=False,
            )

            cv2.destroyWindow(
                "Manual Bounding Box"
            )

            x, y, w, h = roi

            if w > 0 and h > 0:
                bbox = (
                    int(x / scale),
                    int(y / scale),
                    int(w / scale),
                    int(h / scale),
                )

        # S = skip
        elif key in (ord("s"), ord("S")):
            return "skipped"

        # Q = quit
        elif key in (ord("q"), ord("Q")):
            return "quit"


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("BASLER THREE-CLASS YOLO ANNOTATOR")
    print("=" * 65)

    total_images = 0
    total_existing = 0
    total_accepted = 0
    total_skipped = 0

    for class_name, class_id in CLASS_INFO.items():

        folder = DATASET_ROOT / class_name

        if not folder.exists():
            print(
                f"\nWARNING: folder does not exist: {folder}"
            )
            continue

        images = sorted([
            p
            for p in folder.iterdir()
            if p.suffix.lower() in IMAGE_EXTENSIONS
        ])

        print()
        print(
            f"{class_name.upper()}: "
            f"{len(images)} images "
            f"(class {class_id})"
        )

        for index, image_path in enumerate(
            images,
            start=1,
        ):
            total_images += 1

            print(
                f"[{index}/{len(images)}] "
                f"{image_path.name}"
            )

            result = annotate_image(
                image_path,
                class_name,
                class_id,
            )

            if result == "existing":
                total_existing += 1

            elif result == "accepted":
                total_accepted += 1

            elif result == "skipped":
                total_skipped += 1

            elif result == "quit":
                cv2.destroyAllWindows()

                print("\nStopped.")
                return

    cv2.destroyAllWindows()

    print()
    print("=" * 65)
    print("ANNOTATION COMPLETE")
    print("=" * 65)

    print(f"Images seen       : {total_images}")
    print(f"New annotations   : {total_accepted}")
    print(f"Already annotated : {total_existing}")
    print(f"Skipped           : {total_skipped}")


if __name__ == "__main__":
    main()