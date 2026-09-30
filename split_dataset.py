from pathlib import Path
import random
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

CIRCLE_FOLDER = Path(
    r"H:\ABB_Vision_Project\basler_raw\circle_raw"
)

SQUARE_FOLDER = Path(
    r"H:\ABB_Vision_Project\basler_raw\square_raw"
)

TRIANGLE_FOLDER = Path(
    r"H:\ABB_Vision_Project\basler_raw\triangle_raw"
)

OUTPUT_FOLDER = Path(
    r"H:\ABB_Vision_Project\dataset_3class"
)


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


CLASS_INFO = {
    "square": {
        "id": 0,
        "folder": SQUARE_FOLDER,
    },

    "circle": {
        "id": 1,
        "folder": CIRCLE_FOLDER,
    },

    "triangle": {
        "id": 2,
        "folder": TRIANGLE_FOLDER,
    },
}


# ============================================================
# VALIDATE RATIOS
# ============================================================

def validate_ratios():

    total = (
        TRAIN_RATIO
        + VAL_RATIO
        + TEST_RATIO
    )

    if abs(total - 1.0) > 1e-9:

        raise ValueError(
            "TRAIN_RATIO + VAL_RATIO + TEST_RATIO "
            f"must equal 1.0, but got {total}"
        )


# ============================================================
# FIND IMAGE/LABEL PAIRS
# ============================================================

def find_pairs(folder):
    """
    Find supported images and their corresponding YOLO labels.

    Returns:
        [
            (image_path, label_path),
            ...
        ]

    Raises an error if an image does not have a matching label.
    """

    if not folder.exists():

        raise FileNotFoundError(
            f"Folder not found:\n{folder}"
        )

    images = sorted([
        path
        for path in folder.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ])

    pairs = []

    missing_labels = []

    for image_path in images:

        label_path = image_path.with_suffix(".txt")

        if not label_path.exists():

            missing_labels.append(
                image_path
            )

            continue

        pairs.append(
            (
                image_path,
                label_path,
            )
        )

    # --------------------------------------------------------
    # Fail immediately if anything is unannotated
    # --------------------------------------------------------

    if missing_labels:

        print()
        print("=" * 70)
        print("ERROR: MISSING LABELS")
        print("=" * 70)

        for path in missing_labels:

            print(
                f"No label for: {path.name}"
            )

        raise RuntimeError(
            f"{len(missing_labels)} images "
            "do not have matching YOLO labels."
        )

    return pairs


# ============================================================
# VERIFY LABEL CLASS
# ============================================================

def verify_labels(
    pairs,
    expected_class_id,
    class_name,
):
    """
    Verify that every annotation belongs to the expected class.

    Example:
        square -> class 0
        circle -> class 1
        triangle -> class 2

    This dataset currently expects exactly one target
    annotation per image.
    """

    errors = []

    for image_path, label_path in pairs:

        text = label_path.read_text(
            encoding="utf-8"
        ).strip()

        if not text:

            errors.append(
                f"{label_path.name}: empty label"
            )

            continue

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        # We expect one object per image in this dataset.
        if len(lines) != 1:

            errors.append(
                f"{label_path.name}: expected 1 annotation, "
                f"found {len(lines)}"
            )

            continue

        parts = lines[0].split()

        if len(parts) != 5:

            errors.append(
                f"{label_path.name}: "
                "invalid YOLO format"
            )

            continue

        try:

            class_id = int(parts[0])

            values = [
                float(value)
                for value in parts[1:]
            ]

        except ValueError:

            errors.append(
                f"{label_path.name}: "
                "contains invalid numbers"
            )

            continue

        if class_id != expected_class_id:

            errors.append(
                f"{label_path.name}: "
                f"expected class {expected_class_id} "
                f"({class_name}), "
                f"found class {class_id}"
            )

        cx, cy, width, height = values

        # Check normalized YOLO coordinates
        if not (
            0 <= cx <= 1
            and 0 <= cy <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):

            errors.append(
                f"{label_path.name}: "
                "YOLO coordinates outside valid range"
            )

        # Check bounding box stays inside image
        left = cx - width / 2
        right = cx + width / 2

        top = cy - height / 2
        bottom = cy + height / 2

        tolerance = 1e-4

        if (
            left < -tolerance
            or right > 1 + tolerance
            or top < -tolerance
            or bottom > 1 + tolerance
        ):

            errors.append(
                f"{label_path.name}: "
                "bounding box extends outside image"
            )

    if errors:

        print()
        print("=" * 70)
        print(
            f"LABEL ERRORS: {class_name.upper()}"
        )
        print("=" * 70)

        for error in errors:

            print(error)

        raise RuntimeError(
            f"{len(errors)} label errors found "
            f"in {class_name}."
        )


# ============================================================
# SPLIT ONE CLASS
# ============================================================

def split_pairs(pairs, rng):
    """
    Split image/label pairs into:

        train
        validation
        test

    The image and its annotation always stay together.
    """

    pairs = pairs.copy()

    rng.shuffle(pairs)

    total = len(pairs)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    # Everything remaining goes into test.
    # This prevents images from being lost because
    # of integer rounding.
    test_count = (
        total
        - train_count
        - val_count
    )

    train = pairs[
        :train_count
    ]

    val = pairs[
        train_count:
        train_count + val_count
    ]

    test = pairs[
        train_count + val_count:
    ]

    assert (
        len(train)
        + len(val)
        + len(test)
        == total
    )

    assert len(test) == test_count

    return train, val, test


# ============================================================
# COPY IMAGE + LABEL
# ============================================================

def copy_pairs(
    pairs,
    image_destination,
    label_destination,
    prefix,
):
    """
    Copy images and labels together.

    Example output:

        images/train/square_0001.jpg
        labels/train/square_0001.txt
    """

    image_destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    for number, (
        image_source,
        label_source,
    ) in enumerate(
        pairs,
        start=1,
    ):

        stem = (
            f"{prefix}_{number:04d}"
        )

        image_name = (
            stem
            + image_source.suffix.lower()
        )

        label_name = (
            stem
            + ".txt"
        )

        image_target = (
            image_destination
            / image_name
        )

        label_target = (
            label_destination
            / label_name
        )

        shutil.copy2(
            image_source,
            image_target,
        )

        shutil.copy2(
            label_source,
            label_target,
        )


# ============================================================
# CREATE OUTPUT STRUCTURE
# ============================================================

def create_output_directories():

    for split in [
        "train",
        "val",
        "test",
    ]:

        (
            OUTPUT_FOLDER
            / "images"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            OUTPUT_FOLDER
            / "labels"
            / split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


# ============================================================
# CREATE DATA.YAML
# ============================================================

def create_yaml():

    yaml_content = """\
path: .

train: images/train
val: images/val
test: images/test

names:
  0: square
  1: circle
  2: triangle
"""

    yaml_path = (
        OUTPUT_FOLDER
        / "data.yaml"
    )

    yaml_path.write_text(
        yaml_content,
        encoding="utf-8",
    )


# ============================================================
# COUNT OUTPUT
# ============================================================

def count_output(split):

    image_folder = (
        OUTPUT_FOLDER
        / "images"
        / split
    )

    label_folder = (
        OUTPUT_FOLDER
        / "labels"
        / split
    )

    images = [
        path
        for path in image_folder.iterdir()
        if path.suffix.lower()
        in SUPPORTED_EXTENSIONS
    ]

    labels = list(
        label_folder.glob("*.txt")
    )

    return (
        len(images),
        len(labels),
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "INA700 THREE-CLASS DATASET SPLITTER"
    )
    print("=" * 70)

    validate_ratios()

    rng = random.Random(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Read and verify all classes BEFORE deleting/creating
    # anything.
    # --------------------------------------------------------

    datasets = {}

    print()
    print("Checking source dataset...")

    for class_name, info in CLASS_INFO.items():

        pairs = find_pairs(
            info["folder"]
        )

        if not pairs:

            raise RuntimeError(
                f"No {class_name} images found."
            )

        verify_labels(
            pairs,
            info["id"],
            class_name,
        )

        datasets[class_name] = pairs

        print(
            f"{class_name.capitalize():<10}: "
            f"{len(pairs)} image/label pairs"
        )

    # --------------------------------------------------------
    # Check expected total
    # --------------------------------------------------------

    total_source = sum(
        len(pairs)
        for pairs in datasets.values()
    )

    print()
    print(
        f"Total source pairs: {total_source}"
    )

    # --------------------------------------------------------
    # Remove OLD generated output only
    # --------------------------------------------------------

    if OUTPUT_FOLDER.exists():

        print()
        print(
            "Removing previous generated dataset:"
        )

        print(
            OUTPUT_FOLDER
        )

        shutil.rmtree(
            OUTPUT_FOLDER
        )

    create_output_directories()

    # --------------------------------------------------------
    # Split each class separately
    # --------------------------------------------------------

    split_results = {}

    for class_name, pairs in datasets.items():

        train, val, test = split_pairs(
            pairs,
            rng,
        )

        split_results[class_name] = {
            "train": train,
            "val": val,
            "test": test,
        }

    # --------------------------------------------------------
    # Copy everything
    # --------------------------------------------------------

    for class_name, splits in split_results.items():

        for split_name, pairs in splits.items():

            copy_pairs(
                pairs,

                OUTPUT_FOLDER
                / "images"
                / split_name,

                OUTPUT_FOLDER
                / "labels"
                / split_name,

                class_name,
            )

    # --------------------------------------------------------
    # YAML
    # --------------------------------------------------------

    create_yaml()

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DATASET CREATED")
    print("=" * 70)

    grand_train = 0
    grand_val = 0
    grand_test = 0

    for class_name in [
        "square",
        "circle",
        "triangle",
    ]:

        results = split_results[
            class_name
        ]

        train_count = len(
            results["train"]
        )

        val_count = len(
            results["val"]
        )

        test_count = len(
            results["test"]
        )

        grand_train += train_count
        grand_val += val_count
        grand_test += test_count

        print()
        print(
            class_name.upper()
        )

        print(
            f"  Train : {train_count}"
        )

        print(
            f"  Val   : {val_count}"
        )

        print(
            f"  Test  : {test_count}"
        )

        print(
            f"  Total : "
            f"{train_count + val_count + test_count}"
        )

    print()
    print("-" * 70)

    print(
        f"TOTAL TRAIN : {grand_train}"
    )

    print(
        f"TOTAL VAL   : {grand_val}"
    )

    print(
        f"TOTAL TEST  : {grand_test}"
    )

    print(
        f"TOTAL       : "
        f"{grand_train + grand_val + grand_test}"
    )

    # --------------------------------------------------------
    # Final integrity check
    # --------------------------------------------------------

    print()
    print("OUTPUT INTEGRITY CHECK")

    for split in [
        "train",
        "val",
        "test",
    ]:

        image_count, label_count = (
            count_output(split)
        )

        status = (
            "OK"
            if image_count == label_count
            else "ERROR"
        )

        print(
            f"  {split:<5}: "
            f"{image_count} images, "
            f"{label_count} labels "
            f"[{status}]"
        )

        if image_count != label_count:

            raise RuntimeError(
                f"Image/label mismatch "
                f"in {split}."
            )

    print()
    print(
        f"Dataset location:\n"
        f"{OUTPUT_FOLDER.resolve()}"
    )

    print()
    print(
        "Original Basler images and labels "
        "were NOT modified."
    )

    print()
    print(
        "Class mapping:"
    )

    print(
        "  0 = square"
    )

    print(
        "  1 = circle"
    )

    print(
        "  2 = triangle"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()