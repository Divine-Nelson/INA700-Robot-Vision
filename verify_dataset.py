from pathlib import Path
import random
import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = Path("dataset_3class")

SPLITS = ["train", "val"]

CLASS_NAMES = {
    0: "square",
    1: "circle",
    2: "triangle"
}

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

# Number of random images on each contact sheet
SAMPLES_PER_SPLIT = 20

RANDOM_SEED = 42

OUTPUT_FOLDER = Path("verification_results")


# ============================================================
# READ LABEL
# ============================================================

def read_yolo_label(label_path):

    annotations = []

    try:
        lines = label_path.read_text(
            encoding="utf-8"
        ).splitlines()

    except Exception as error:
        return None, [
            f"Could not read label: {error}"
        ]

    errors = []

    for line_number, line in enumerate(
        lines,
        start=1
    ):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:

            errors.append(
                f"Line {line_number}: "
                f"expected 5 values, got {len(parts)}"
            )

            continue

        try:

            class_id = int(parts[0])

            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

        except ValueError:

            errors.append(
                f"Line {line_number}: "
                f"contains invalid numbers"
            )

            continue

        annotations.append({
            "class_id": class_id,
            "cx": cx,
            "cy": cy,
            "w": w,
            "h": h,
            "line": line_number,
        })

    return annotations, errors


# ============================================================
# VALIDATE ANNOTATION
# ============================================================

def validate_annotation(annotation):

    errors = []

    class_id = annotation["class_id"]

    cx = annotation["cx"]
    cy = annotation["cy"]
    w = annotation["w"]
    h = annotation["h"]

    line = annotation["line"]

    # --------------------------------------------------------
    # Class ID
    # --------------------------------------------------------

    if class_id not in CLASS_NAMES:

        errors.append(
            f"Line {line}: "
            f"invalid class ID {class_id}"
        )

    # --------------------------------------------------------
    # Basic normalized ranges
    # --------------------------------------------------------

    if not 0 <= cx <= 1:

        errors.append(
            f"Line {line}: "
            f"center_x outside 0-1: {cx}"
        )

    if not 0 <= cy <= 1:

        errors.append(
            f"Line {line}: "
            f"center_y outside 0-1: {cy}"
        )

    if not 0 < w <= 1:

        errors.append(
            f"Line {line}: "
            f"width invalid: {w}"
        )

    if not 0 < h <= 1:

        errors.append(
            f"Line {line}: "
            f"height invalid: {h}"
        )

    # --------------------------------------------------------
    # Check box boundaries
    # --------------------------------------------------------

    left = cx - w / 2
    right = cx + w / 2

    top = cy - h / 2
    bottom = cy + h / 2

    tolerance = 0.00001

    if left < -tolerance:
        errors.append(
            f"Line {line}: "
            f"box extends beyond LEFT edge"
        )

    if right > 1 + tolerance:
        errors.append(
            f"Line {line}: "
            f"box extends beyond RIGHT edge"
        )

    if top < -tolerance:
        errors.append(
            f"Line {line}: "
            f"box extends beyond TOP edge"
        )

    if bottom > 1 + tolerance:
        errors.append(
            f"Line {line}: "
            f"box extends beyond BOTTOM edge"
        )

    return errors


# ============================================================
# FIND IMAGES
# ============================================================

def find_images(folder):

    return sorted([
        path
        for path in folder.iterdir()
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ])


# ============================================================
# VERIFY SPLIT
# ============================================================

def verify_split(split):

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

    images = find_images(
        image_folder
    )

    labels = sorted(
        label_folder.glob("*.txt")
    )

    image_stems = {
        image.stem
        for image in images
    }

    label_stems = {
        label.stem
        for label in labels
    }

    missing_labels = (
        image_stems -
        label_stems
    )

    orphan_labels = (
        label_stems -
        image_stems
    )

    errors = []

    class_counts = {
        0: 0,
        1: 0,
    }

    unreadable_images = []

    empty_labels = []

    # --------------------------------------------------------
    # Verify images
    # --------------------------------------------------------

    for image_path in images:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            unreadable_images.append(
                image_path.name
            )

    # --------------------------------------------------------
    # Verify labels
    # --------------------------------------------------------

    for label_path in labels:

        annotations, read_errors = (
            read_yolo_label(
                label_path
            )
        )

        for error in read_errors:

            errors.append(
                f"{label_path.name}: {error}"
            )

        if annotations is None:
            continue

        if len(annotations) == 0:

            empty_labels.append(
                label_path.name
            )

        for annotation in annotations:

            annotation_errors = (
                validate_annotation(
                    annotation
                )
            )

            for error in annotation_errors:

                errors.append(
                    f"{label_path.name}: "
                    f"{error}"
                )

            class_id = (
                annotation["class_id"]
            )

            if class_id in class_counts:

                class_counts[
                    class_id
                ] += 1

    return {
        "images": images,
        "labels": labels,
        "missing_labels": sorted(
            missing_labels
        ),
        "orphan_labels": sorted(
            orphan_labels
        ),
        "errors": errors,
        "class_counts": class_counts,
        "unreadable_images": (
            unreadable_images
        ),
        "empty_labels": empty_labels,
    }


# ============================================================
# DRAW YOLO BOXES
# ============================================================

def draw_annotations(
    image,
    annotations
):

    output = image.copy()

    image_height, image_width = (
        image.shape[:2]
    )

    for annotation in annotations:

        class_id = (
            annotation["class_id"]
        )

        cx = (
            annotation["cx"] *
            image_width
        )

        cy = (
            annotation["cy"] *
            image_height
        )

        w = (
            annotation["w"] *
            image_width
        )

        h = (
            annotation["h"] *
            image_height
        )

        x1 = int(
            cx - w / 2
        )

        y1 = int(
            cy - h / 2
        )

        x2 = int(
            cx + w / 2
        )

        y2 = int(
            cy + h / 2
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

        class_name = (
            CLASS_NAMES.get(
                class_id,
                f"unknown_{class_id}"
            )
        )

        cv2.putText(
            output,
            class_name.upper(),
            (
                x1,
                max(y1 - 10, 25)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    return output


# ============================================================
# CREATE CONTACT SHEET
# ============================================================

def create_contact_sheet(
    split,
    images,
):

    if not images:
        return None

    random.seed(
        RANDOM_SEED +
        (0 if split == "train" else 1)
    )

    sample_count = min(
        SAMPLES_PER_SPLIT,
        len(images)
    )

    selected = random.sample(
        images,
        sample_count
    )

    # Each preview tile
    tile_width = 320
    tile_height = 260

    image_height = 220

    columns = 4

    rows = int(
        np.ceil(
            sample_count /
            columns
        )
    )

    sheet = np.zeros(
        (
            rows * tile_height,
            columns * tile_width,
            3
        ),
        dtype=np.uint8
    )

    for index, image_path in enumerate(
        selected
    ):

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        label_path = (
            DATASET /
            "labels" /
            split /
            f"{image_path.stem}.txt"
        )

        annotations = []

        if label_path.exists():

            annotations, _ = (
                read_yolo_label(
                    label_path
                )
            )

            if annotations is None:
                annotations = []

        annotated = draw_annotations(
            image,
            annotations
        )

        # ----------------------------------------------------
        # Resize while preserving aspect ratio
        # ----------------------------------------------------

        h, w = annotated.shape[:2]

        scale = min(
            tile_width / w,
            image_height / h
        )

        resized_width = int(
            w * scale
        )

        resized_height = int(
            h * scale
        )

        resized = cv2.resize(
            annotated,
            (
                resized_width,
                resized_height
            ),
            interpolation=cv2.INTER_AREA
        )

        # ----------------------------------------------------
        # Tile
        # ----------------------------------------------------

        tile = np.zeros(
            (
                tile_height,
                tile_width,
                3
            ),
            dtype=np.uint8
        )

        x_offset = (
            tile_width -
            resized_width
        ) // 2

        tile[
            0:resized_height,
            x_offset:
            x_offset + resized_width
        ] = resized

        # Filename
        cv2.putText(
            tile,
            image_path.name[:38],
            (
                8,
                tile_height - 12
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (255, 255, 255),
            1
        )

        row = (
            index //
            columns
        )

        column = (
            index %
            columns
        )

        y1 = (
            row *
            tile_height
        )

        x1 = (
            column *
            tile_width
        )

        sheet[
            y1:
            y1 + tile_height,
            x1:
            x1 + tile_width
        ] = tile

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        OUTPUT_FOLDER /
        f"{split}_contact_sheet.jpg"
    )

    cv2.imwrite(
        str(output_path),
        sheet
    )

    return output_path


# ============================================================
# PRINT REPORT
# ============================================================

def print_report(
    split,
    result
):

    print()
    print("=" * 60)
    print(
        f"{split.upper()} VERIFICATION"
    )
    print("=" * 60)

    print(
        f"Images: "
        f"{len(result['images'])}"
    )

    print(
        f"Label files: "
        f"{len(result['labels'])}"
    )

    print()

    print(
        f"Square annotations: "
        f"{result['class_counts'][0]}"
    )

    print(
        f"Circle annotations: "
        f"{result['class_counts'][1]}"
    )
    print(
        f"Triangle annotations: "
        f"{result['class_counts'][1]}"
    )

    print()

    print(
        f"Missing labels: "
        f"{len(result['missing_labels'])}"
    )

    print(
        f"Orphan labels: "
        f"{len(result['orphan_labels'])}"
    )

    print(
        f"Unreadable images: "
        f"{len(result['unreadable_images'])}"
    )

    print(
        f"Empty labels: "
        f"{len(result['empty_labels'])}"
    )

    print(
        f"Annotation errors: "
        f"{len(result['errors'])}"
    )

    # --------------------------------------------------------
    # Detailed problems
    # --------------------------------------------------------

    if result["missing_labels"]:

        print()
        print("MISSING LABELS:")

        for name in result[
            "missing_labels"
        ]:

            print(
                f"  - {name}"
            )

    if result["orphan_labels"]:

        print()
        print("ORPHAN LABELS:")

        for name in result[
            "orphan_labels"
        ]:

            print(
                f"  - {name}"
            )

    if result["unreadable_images"]:

        print()
        print("UNREADABLE IMAGES:")

        for name in result[
            "unreadable_images"
        ]:

            print(
                f"  - {name}"
            )

    if result["errors"]:

        print()
        print("ANNOTATION ERRORS:")

        for error in result[
            "errors"
        ]:

            print(
                f"  - {error}"
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 60)
    print(
        "INA700 YOLO DATASET VERIFICATION"
    )
    print("=" * 60)

    all_good = True

    for split in SPLITS:

        result = verify_split(
            split
        )

        print_report(
            split,
            result
        )

        contact_sheet = (
            create_contact_sheet(
                split,
                result["images"]
            )
        )

        if contact_sheet:

            print()
            print(
                f"Contact sheet: "
                f"{contact_sheet}"
            )

        serious_problem = any([
            result["missing_labels"],
            result["orphan_labels"],
            result["unreadable_images"],
            result["errors"],
        ])

        if serious_problem:
            all_good = False

    print()
    print("=" * 60)

    if all_good:

        print(
            "STRUCTURAL VERIFICATION PASSED"
        )

        print()
        print(
            "Now visually inspect the "
            "generated contact sheets."
        )

    else:

        print(
            "VERIFICATION FOUND PROBLEMS"
        )

        print()
        print(
            "Fix the issues above before training."
        )

    print("=" * 60)


if __name__ == "__main__":
    main()