import cv2
from pathlib import Path
import shutil

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"H:\ABB_Vision_Project")
SOURCE_ROOT = PROJECT_ROOT / "basler_multi_rawimages"

# We will NOT overwrite the original single-object labels.
OUTPUT_ROOT = PROJECT_ROOT / "basler_combine_labels"

CLASS_NAMES = {
    0: "square",
    1: "circle",
    2: "triangle",
}

CLASS_KEYS = {
    ord("0"): 0,
    ord("1"): 1,
    ord("2"): 2,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

MAX_PREVIEW_WIDTH = 1200
MAX_PREVIEW_HEIGHT = 800


# ============================================================
# GLOBAL GUI STATE
# ============================================================

drawing = False
start_point = None
current_point = None

boxes = []
# Each item:
# {
#     "class_id": 1,
#     "bbox": (x1, y1, x2, y2)
# }


# ============================================================
# HELPERS
# ============================================================

"""def get_image_files():
    images = []

    for folder in SOURCE_ROOT.iterdir():
        if not folder.is_dir():
            continue

        for path in folder.iterdir():
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append(path)

    return sorted(images)
"""

def get_image_files():
    return sorted(
        path
        for path in SOURCE_ROOT.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

def get_output_label_path(image_path):
    relative = image_path.relative_to(SOURCE_ROOT)

    output_dir = OUTPUT_ROOT / relative.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    return (OUTPUT_ROOT / relative).with_suffix(".txt")


def calculate_scale(image):
    h, w = image.shape[:2]

    scale_w = MAX_PREVIEW_WIDTH / w
    scale_h = MAX_PREVIEW_HEIGHT / h

    return min(scale_w, scale_h, 1.0)


def to_original_coordinates(x, y, scale):
    return int(x / scale), int(y / scale)


def to_preview_coordinates(x, y, scale):
    return int(x * scale), int(y * scale)


def normalize_bbox(x1, y1, x2, y2, image_width, image_height):
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    x_center = ((x1 + x2) / 2.0) / image_width
    y_center = ((y1 + y2) / 2.0) / image_height

    width = (x2 - x1) / image_width
    height = (y2 - y1) / image_height

    return x_center, y_center, width, height


def save_annotations(image_path, image):
    label_path = get_output_label_path(image_path)

    h, w = image.shape[:2]

    lines = []

    for item in boxes:
        class_id = item["class_id"]
        x1, y1, x2, y2 = item["bbox"]

        xc, yc, bw, bh = normalize_bbox(
            x1, y1, x2, y2, w, h
        )

        lines.append(
            f"{class_id} "
            f"{xc:.6f} "
            f"{yc:.6f} "
            f"{bw:.6f} "
            f"{bh:.6f}"
        )

    label_path.write_text("\n".join(lines), encoding="utf-8")

    return label_path


def load_existing_annotations(image_path, image):
    global boxes

    boxes = []

    label_path = get_output_label_path(image_path)

    if not label_path.exists():
        return

    h, w = image.shape[:2]

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()

        if len(parts) != 5:
            continue

        class_id = int(parts[0])

        xc = float(parts[1]) * w
        yc = float(parts[2]) * h
        bw = float(parts[3]) * w
        bh = float(parts[4]) * h

        x1 = int(xc - bw / 2)
        y1 = int(yc - bh / 2)
        x2 = int(xc + bw / 2)
        y2 = int(yc + bh / 2)

        boxes.append({
            "class_id": class_id,
            "bbox": (x1, y1, x2, y2),
        })


def draw_annotations(image, scale):
    preview = cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )

    # Existing boxes
    for i, item in enumerate(boxes):
        class_id = item["class_id"]
        x1, y1, x2, y2 = item["bbox"]

        px1, py1 = to_preview_coordinates(x1, y1, scale)
        px2, py2 = to_preview_coordinates(x2, y2, scale)

        cv2.rectangle(
            preview,
            (px1, py1),
            (px2, py2),
            (0, 255, 0),
            2,
        )

        label = f"{i + 1}: {CLASS_NAMES[class_id]}"

        cv2.putText(
            preview,
            label,
            (px1, max(py1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    # Box currently being drawn
    if drawing and start_point is not None and current_point is not None:
        cv2.rectangle(
            preview,
            start_point,
            current_point,
            (0, 255, 255),
            2,
        )

    return preview


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global drawing
    global start_point
    global current_point

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_point = (x, y)


# ============================================================
# ANNOTATE ONE IMAGE
# ============================================================

def annotate_image(image_path):
    global boxes
    global start_point
    global current_point
    global drawing

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read: {image_path}")
        return "next"

    scale = calculate_scale(image)

    load_existing_annotations(image_path, image)

    drawing = False
    start_point = None
    current_point = None

    window_name = "Multi-object YOLO Annotator"

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print()
    print("=" * 70)
    print(image_path)
    print()
    print("DRAW box with LEFT MOUSE")
    print("After drawing:")
    print("  0 = square")
    print("  1 = circle")
    print("  2 = triangle")
    print()
    print("Other controls:")
    print("  U = undo last box")
    print("  S = save + next image")
    print("  B = previous image")
    print("  Q = save + quit")
    print("=" * 70)

    pending_box = None

    while True:
        display = draw_annotations(image, scale)

        cv2.putText(
            display,
            f"Objects: {len(boxes)}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
        )

        cv2.imshow(window_name, display)

        key = cv2.waitKey(20) & 0xFF

        # Mouse drag finished?
        if (
            not drawing
            and start_point is not None
            and current_point is not None
            and pending_box is None
        ):
            sx, sy = start_point
            ex, ey = current_point

            if abs(ex - sx) > 5 and abs(ey - sy) > 5:
                x1, y1 = to_original_coordinates(sx, sy, scale)
                x2, y2 = to_original_coordinates(ex, ey, scale)

                pending_box = (
                    min(x1, x2),
                    min(y1, y2),
                    max(x1, x2),
                    max(y1, y2),
                )

                print()
                print("Box drawn.")
                print("Press 0=Square, 1=Circle, 2=Triangle")

            start_point = None
            current_point = None

        # Assign class to pending box
        if pending_box is not None and key in CLASS_KEYS:
            class_id = CLASS_KEYS[key]

            boxes.append({
                "class_id": class_id,
                "bbox": pending_box,
            })

            print(
                f"Added: {CLASS_NAMES[class_id]} "
                f"(total objects: {len(boxes)})"
            )

            pending_box = None

        # Undo
        elif key in (ord("u"), ord("U")):
            if boxes:
                removed = boxes.pop()
                print(
                    f"Removed: "
                    f"{CLASS_NAMES[removed['class_id']]}"
                )

            pending_box = None

        # Save + next
        elif key in (ord("s"), ord("S")):
            label_path = save_annotations(
                image_path,
                image,
            )

            print(
                f"Saved {len(boxes)} objects -> "
                f"{label_path}"
            )

            cv2.destroyWindow(window_name)
            return "next"

        # Previous image
        elif key in (ord("b"), ord("B")):
            save_annotations(image_path, image)

            cv2.destroyWindow(window_name)
            return "previous"

        # Quit
        elif key in (ord("q"), ord("Q")):
            save_annotations(image_path, image)

            cv2.destroyAllWindows()
            return "quit"


# ============================================================
# MAIN
# ============================================================

def main():
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(
            f"Source dataset not found:\n{SOURCE_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = get_image_files()

    if not images:
        raise RuntimeError(
            f"No images found under:\n{SOURCE_ROOT}"
        )

    print()
    print("=" * 70)
    print("MULTI-OBJECT YOLO ANNOTATOR")
    print("=" * 70)
    print(f"Images found: {len(images)}")
    print(f"Source: {SOURCE_ROOT}")
    print(f"Labels: {OUTPUT_ROOT}")
    print()
    print("Class mapping:")
    print("0 = square")
    print("1 = circle")
    print("2 = triangle")
    print("=" * 70)

    index = 0

    while 0 <= index < len(images):
        print(
            f"\nImage {index + 1}/{len(images)}"
        )

        result = annotate_image(images[index])

        if result == "quit":
            break

        elif result == "previous":
            index = max(0, index - 1)

        else:
            index += 1

    cv2.destroyAllWindows()

    print()
    print("Annotation session finished.")
    print(
        f"Labels stored safely in:\n"
        f"{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()