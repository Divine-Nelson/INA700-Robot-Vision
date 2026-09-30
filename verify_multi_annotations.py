from pathlib import Path
from collections import Counter
import math

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"H:\ABB_Vision_Project")

# New V3 targeted / combination images
IMAGE_ROOT = PROJECT_ROOT / "basler_multi_rawimages"

# Labels for the new V3 targeted images
LABEL_ROOT = PROJECT_ROOT / "basler_combine_labels"

# Verification output
OUTPUT_ROOT = PROJECT_ROOT / "combine_verification"


CLASS_NAMES = {
    0: "square",
    1: "circle",
    2: "triangle",
}


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# Contact-sheet settings
CONTACT_SHEET_LIMIT = 100
THUMB_WIDTH = 320
THUMB_HEIGHT = 260
COLUMNS = 4


# ============================================================
# FILE HELPERS
# ============================================================

def find_images():
    """
    Find all supported images recursively under IMAGE_ROOT.

    This works whether images are stored directly inside
    basler_multi_rawimages or inside subfolders.
    """

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"Image root does not exist:\n{IMAGE_ROOT}"
        )

    images = []

    for path in IMAGE_ROOT.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            images.append(path)

    return sorted(images)


def label_path_for(image_path):
    """
    Return the corresponding YOLO label path.

    Example:

        basler_multi_rawimages/combine_0001.jpg

    becomes:

        basler_combine_labels/combine_0001.txt

    Subfolder structure is preserved automatically.
    """

    relative = image_path.relative_to(IMAGE_ROOT)

    return (
        LABEL_ROOT / relative
    ).with_suffix(".txt")


# ============================================================
# YOLO LABEL READER / VALIDATOR
# ============================================================

def read_annotations(label_path):
    """
    Read and validate a YOLO detection label file.

    Format:
        class_id x_center y_center width height

    IMPORTANT:
    An existing EMPTY .txt file is VALID.

    It represents a negative/background image containing
    no target objects.

    Example:
        image contains only hexagon + star
        -> label file should intentionally be empty.
    """

    annotations = []
    errors = []

    # --------------------------------------------------------
    # Missing label is an error.
    # --------------------------------------------------------

    if not label_path.exists():
        return annotations, ["missing label"]

    # --------------------------------------------------------
    # Read label file.
    # --------------------------------------------------------

    try:
        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

    except Exception as exc:
        return annotations, [
            f"could not read label: {exc}"
        ]

    # --------------------------------------------------------
    # Empty label is VALID.
    # This is an intentional negative/background image.
    # --------------------------------------------------------

    if not lines:
        return annotations, []

    # --------------------------------------------------------
    # Validate each annotation.
    # --------------------------------------------------------

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        line = line.strip()

        # Blank lines inside a non-empty file are ignored.
        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"line {line_number}: "
                f"expected 5 values, got {len(parts)}"
            )
            continue

        # ----------------------------------------------------
        # Parse values
        # ----------------------------------------------------

        try:
            class_id = int(parts[0])

            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

        except ValueError:
            errors.append(
                f"line {line_number}: invalid number"
            )
            continue

        # ----------------------------------------------------
        # Check for NaN / infinity
        # ----------------------------------------------------

        numeric_values = [
            x_center,
            y_center,
            width,
            height,
        ]

        if not all(
            math.isfinite(v)
            for v in numeric_values
        ):
            errors.append(
                f"line {line_number}: "
                f"NaN or infinite coordinate"
            )
            continue

        # ----------------------------------------------------
        # Validate class
        # ----------------------------------------------------

        if class_id not in CLASS_NAMES:
            errors.append(
                f"line {line_number}: "
                f"invalid class {class_id}"
            )
            continue

        # ----------------------------------------------------
        # Validate normalized coordinates
        # ----------------------------------------------------

        if not all(
            0 <= value <= 1
            for value in numeric_values
        ):
            errors.append(
                f"line {line_number}: "
                f"coordinates outside 0..1"
            )
            continue

        # ----------------------------------------------------
        # Validate box dimensions
        # ----------------------------------------------------

        if width <= 0 or height <= 0:
            errors.append(
                f"line {line_number}: "
                f"zero/negative box size"
            )
            continue

        # ----------------------------------------------------
        # Ensure the box does not extend outside the image.
        # ----------------------------------------------------

        x1 = x_center - width / 2
        x2 = x_center + width / 2

        y1 = y_center - height / 2
        y2 = y_center + height / 2

        # Small tolerance for floating-point rounding.
        tolerance = 1e-6

        if (
            x1 < -tolerance
            or y1 < -tolerance
            or x2 > 1 + tolerance
            or y2 > 1 + tolerance
        ):
            errors.append(
                f"line {line_number}: "
                f"box extends outside image"
            )
            continue

        # ----------------------------------------------------
        # Valid annotation
        # ----------------------------------------------------

        annotations.append(
            (
                class_id,
                x_center,
                y_center,
                width,
                height,
            )
        )

    return annotations, errors


# ============================================================
# DRAW ANNOTATIONS
# ============================================================

def draw_annotations(
    image,
    annotations,
    is_negative=False
):
    """
    Draw YOLO boxes on an image.

    Negative/background images receive a visible
    NEGATIVE / BACKGROUND label so they can also be
    inspected on the contact sheet.
    """

    h, w = image.shape[:2]

    output = image.copy()

    # --------------------------------------------------------
    # Draw target annotations
    # --------------------------------------------------------

    for class_id, xc, yc, bw, bh in annotations:

        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)

        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)

        # Clamp only for visualization safety.
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            4,
        )

        label = CLASS_NAMES[class_id]

        cv2.putText(
            output,
            label,
            (x1, max(y1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            3,
        )

    # --------------------------------------------------------
    # Mark intentional negative/background images.
    # --------------------------------------------------------

    if is_negative:

        cv2.putText(
            output,
            "NEGATIVE / BACKGROUND",
            (30, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 0, 255),
            3,
        )

        cv2.putText(
            output,
            "No target objects",
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2,
        )

    return output


# ============================================================
# CONTACT-SHEET HELPERS
# ============================================================

def make_thumbnail(
    image,
    filename,
    object_count
):
    """
    Resize an annotated image and place it on a fixed-size
    canvas for the contact sheet.
    """

    canvas = np.zeros(
        (
            THUMB_HEIGHT,
            THUMB_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    h, w = image.shape[:2]

    # Leave room at the top for filename / object count.
    available_height = THUMB_HEIGHT - 45

    scale = min(
        THUMB_WIDTH / w,
        available_height / h,
    )

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA,
    )

    x_offset = (
        THUMB_WIDTH - new_w
    ) // 2

    y_offset = 45 + (
        available_height - new_h
    ) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    # Filename
    cv2.putText(
        canvas,
        filename[:32],
        (5, 17),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
    )

    # Object count
    cv2.putText(
        canvas,
        f"Targets: {object_count}",
        (5, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
    )

    return canvas


def save_contact_sheet(
    items,
    output_path
):
    """
    Save a collection of thumbnails as one large image.
    """

    if not items:
        return

    rows = (
        len(items) + COLUMNS - 1
    ) // COLUMNS

    sheet = np.zeros(
        (
            rows * THUMB_HEIGHT,
            COLUMNS * THUMB_WIDTH,
            3,
        ),
        dtype=np.uint8,
    )

    for i, item in enumerate(items):

        row = i // COLUMNS
        col = i % COLUMNS

        y1 = row * THUMB_HEIGHT
        y2 = y1 + THUMB_HEIGHT

        x1 = col * THUMB_WIDTH
        x2 = x1 + THUMB_WIDTH

        sheet[
            y1:y2,
            x1:x2
        ] = item

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output_path),
        sheet,
    )


# ============================================================
# MAIN VERIFICATION
# ============================================================

def main():

    # --------------------------------------------------------
    # Validate required folders
    # --------------------------------------------------------

    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(
            f"Image folder not found:\n"
            f"{IMAGE_ROOT}"
        )

    if not LABEL_ROOT.exists():
        raise FileNotFoundError(
            f"Label folder not found:\n"
            f"{LABEL_ROOT}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images = find_images()

    print()
    print("=" * 70)
    print("MULTI-OBJECT DATASET VERIFICATION")
    print("=" * 70)

    print(f"Image root: {IMAGE_ROOT}")
    print(f"Label root: {LABEL_ROOT}")
    print()
    print(f"Images found: {len(images)}")

    if not images:
        raise RuntimeError(
            f"No images found under:\n"
            f"{IMAGE_ROOT}"
        )

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    class_counts = Counter()

    missing_labels = []
    negative_images = []
    invalid_files = []
    unreadable_images = []

    total_annotations = 0

    image_object_counts = Counter()

    # One contact sheet containing all current combination
    # images. This is better than the old square_raw /
    # circle_raw / triangle_raw grouping because these new
    # images can contain several different classes.
    contact_thumbnails = []

    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    for image_path in images:

        label_path = label_path_for(
            image_path
        )

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            unreadable_images.append(
                str(image_path)
            )

            continue

        # ----------------------------------------------------
        # Missing label
        # ----------------------------------------------------

        if not label_path.exists():

            missing_labels.append(
                str(image_path)
            )

            continue

        # ----------------------------------------------------
        # Determine whether this is an intentional negative.
        #
        # strip() handles:
        #   - completely empty files
        #   - files containing only whitespace/newlines
        # ----------------------------------------------------

        try:
            raw_label_text = label_path.read_text(
                encoding="utf-8"
            )

        except Exception as exc:

            invalid_files.append(
                (
                    str(label_path),
                    [f"could not read label: {exc}"],
                )
            )

            continue

        is_negative = (
            raw_label_text.strip() == ""
        )

        if is_negative:

            negative_images.append(
                str(image_path)
            )

        # ----------------------------------------------------
        # Parse / validate annotations
        # ----------------------------------------------------

        annotations, errors = read_annotations(
            label_path
        )

        if errors:

            invalid_files.append(
                (
                    str(label_path),
                    errors,
                )
            )

        # ----------------------------------------------------
        # Count annotations
        # ----------------------------------------------------

        for annotation in annotations:

            class_id = annotation[0]

            class_counts[class_id] += 1

            total_annotations += 1

        image_object_counts[
            len(annotations)
        ] += 1

        # ----------------------------------------------------
        # Draw annotations for contact sheet
        # ----------------------------------------------------

        annotated = draw_annotations(
            image,
            annotations,
            is_negative=is_negative,
        )

        if (
            len(contact_thumbnails)
            < CONTACT_SHEET_LIMIT
        ):

            thumbnail = make_thumbnail(
                annotated,
                image_path.name,
                len(annotations),
            )

            contact_thumbnails.append(
                thumbnail
            )

    # ========================================================
    # SAVE CONTACT SHEET
    # ========================================================

    contact_sheet_path = (
        OUTPUT_ROOT /
        "combined_contact_sheet.jpg"
    )

    save_contact_sheet(
        contact_thumbnails,
        contact_sheet_path,
    )

    # ========================================================
    # RESULTS
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        f"Images:              "
        f"{len(images)}"
    )

    print(
        f"Total annotations:   "
        f"{total_annotations}"
    )

    print(
        f"Square instances:    "
        f"{class_counts[0]}"
    )

    print(
        f"Circle instances:    "
        f"{class_counts[1]}"
    )

    print(
        f"Triangle instances:  "
        f"{class_counts[2]}"
    )

    print()

    print("Objects per image:")

    for count in sorted(
        image_object_counts
    ):

        print(
            f"  {count} object(s): "
            f"{image_object_counts[count]} images"
        )

    print()

    print(
        f"Missing labels:      "
        f"{len(missing_labels)}"
    )

    print(
        f"Negative images:     "
        f"{len(negative_images)}"
    )

    print(
        f"Unreadable images:   "
        f"{len(unreadable_images)}"
    )

    print(
        f"Invalid label files: "
        f"{len(invalid_files)}"
    )

    # ========================================================
    # DETAILED PROBLEM REPORTS
    # ========================================================

    if missing_labels:

        print()
        print("MISSING LABELS:")

        for item in missing_labels:
            print(item)

    # --------------------------------------------------------
    # Negative images are INFORMATION, not errors.
    # --------------------------------------------------------

    if negative_images:

        print()
        print(
            "VALID NEGATIVE/BACKGROUND IMAGES:"
        )

        for item in negative_images:
            print(item)

    if unreadable_images:

        print()
        print("UNREADABLE IMAGES:")

        for item in unreadable_images:
            print(item)

    if invalid_files:

        print()
        print("INVALID LABELS:")

        for path, errors in invalid_files:

            print(path)

            for error in errors:

                print(
                    f"    {error}"
                )

    # ========================================================
    # STRUCTURAL VERIFICATION
    # ========================================================

    print()
    print("=" * 70)

    # IMPORTANT:
    #
    # We intentionally DO NOT check:
    #
    #     len(images) == 313
    #
    # because 313 belonged to the old V2 source dataset.
    #
    # We also DO NOT fail because negative_images exists.
    # Empty labels are legitimate YOLO background examples.
    #
    # Verification fails only for actual structural problems.

    passed = (
        len(images) > 0
        and not missing_labels
        and not unreadable_images
        and not invalid_files
    )

    if passed:

        print(
            "STRUCTURAL VERIFICATION PASSED"
        )

    else:

        print(
            "STRUCTURAL VERIFICATION FAILED"
        )

    print("=" * 70)

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()
    print(
        "Contact sheet saved to:"
    )

    print(contact_sheet_path)

    print()

    if negative_images:

        print(
            f"NOTE: {len(negative_images)} "
            f"intentional negative/background "
            f"image(s) found."
        )

        print(
            "These are VALID and should remain "
            "in the dataset."
        )

    print()

    print(
        "IMPORTANT: structural verification "
        "does NOT prove the boxes are "
        "semantically correct."
    )

    print(
        "Visually inspect the contact sheet "
        "before training."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()