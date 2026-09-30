from pathlib import Path
import random
import shutil
from collections import Counter

# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"H:\ABB_Vision_Project")

IMAGE_ROOT = PROJECT_ROOT / "basler_raw"
LABEL_ROOT = PROJECT_ROOT / "basler_multi_labels"

OUTPUT_ROOT = PROJECT_ROOT / "dataset_3class_v2"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

SEED = 42

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png",
    ".bmp", ".tif", ".tiff"
}

CLASS_NAMES = {
    0: "square",
    1: "circle",
    2: "triangle",
}


# ============================================================
# FIND IMAGE/LABEL PAIRS
# ============================================================

def find_pairs():

    pairs = []

    for image_path in IMAGE_ROOT.rglob("*"):

        if (
            not image_path.is_file()
            or image_path.suffix.lower()
            not in IMAGE_EXTENSIONS
        ):
            continue

        relative = image_path.relative_to(
            IMAGE_ROOT
        )

        label_path = (
            LABEL_ROOT / relative
        ).with_suffix(".txt")

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label:\n{label_path}"
            )

        if label_path.stat().st_size == 0:
            raise RuntimeError(
                f"Empty label:\n{label_path}"
            )

        pairs.append(
            (image_path, label_path)
        )

    return sorted(
        pairs,
        key=lambda x: str(x[0])
    )


# ============================================================
# READ CLASS IDs
# ============================================================

def read_classes(label_path):

    classes = []

    for line in label_path.read_text(
        encoding="utf-8"
    ).splitlines():

        parts = line.strip().split()

        if len(parts) != 5:
            raise RuntimeError(
                f"Invalid YOLO line in "
                f"{label_path}:\n{line}"
            )

        class_id = int(parts[0])

        if class_id not in CLASS_NAMES:
            raise RuntimeError(
                f"Invalid class {class_id} "
                f"in {label_path}"
            )

        classes.append(class_id)

    return classes


# ============================================================
# SPLIT
# ============================================================

def split_pairs(pairs):

    rng = random.Random(SEED)

    pairs = pairs.copy()
    rng.shuffle(pairs)

    total = len(pairs)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    train = pairs[:train_count]

    val = pairs[
        train_count:
        train_count + val_count
    ]

    test = pairs[
        train_count + val_count:
    ]

    return train, val, test


# ============================================================
# COPY SPLIT
# ============================================================

def copy_split(name, pairs):

    image_output = (
        OUTPUT_ROOT /
        "images" /
        name
    )

    label_output = (
        OUTPUT_ROOT /
        "labels" /
        name
    )

    image_output.mkdir(
        parents=True,
        exist_ok=True
    )

    label_output.mkdir(
        parents=True,
        exist_ok=True
    )

    class_counts = Counter()

    for index, (
        image_path,
        label_path
    ) in enumerate(pairs, start=1):

        # Keep original folder name in filename
        # to avoid collisions and preserve traceability.

        source_folder = (
            image_path.parent.name
        )

        stem = image_path.stem

        new_stem = (
            f"{source_folder}_{stem}"
        )

        destination_image = (
            image_output /
            f"{new_stem}{image_path.suffix.lower()}"
        )

        destination_label = (
            label_output /
            f"{new_stem}.txt"
        )

        shutil.copy2(
            image_path,
            destination_image
        )

        shutil.copy2(
            label_path,
            destination_label
        )

        classes = read_classes(
            label_path
        )

        class_counts.update(
            classes
        )

    return class_counts


# ============================================================
# DATA.YAML
# ============================================================

def write_yaml():

    yaml_content = """path: .

train: images/train
val: images/val
test: images/test

names:
  0: square
  1: circle
  2: triangle
"""

    (
        OUTPUT_ROOT /
        "data.yaml"
    ).write_text(
        yaml_content,
        encoding="utf-8"
    )


# ============================================================
# VERIFY OUTPUT
# ============================================================

def verify_output():

    print()
    print("=" * 70)
    print("OUTPUT INTEGRITY CHECK")
    print("=" * 70)

    for split in [
        "train",
        "val",
        "test",
    ]:

        image_dir = (
            OUTPUT_ROOT /
            "images" /
            split
        )

        label_dir = (
            OUTPUT_ROOT /
            "labels" /
            split
        )

        images = [
            p for p in image_dir.iterdir()
            if p.suffix.lower()
            in IMAGE_EXTENSIONS
        ]

        labels = list(
            label_dir.glob("*.txt")
        )

        image_stems = {
            p.stem for p in images
        }

        label_stems = {
            p.stem for p in labels
        }

        missing = (
            image_stems -
            label_stems
        )

        orphan = (
            label_stems -
            image_stems
        )

        print(
            f"{split.upper():5} "
            f"images={len(images):3} "
            f"labels={len(labels):3} "
            f"missing={len(missing)} "
            f"orphan={len(orphan)}"
        )

        if missing or orphan:
            raise RuntimeError(
                f"{split} integrity "
                f"check failed."
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("CREATE MULTI-OBJECT YOLO DATASET V2")
    print("=" * 70)

    pairs = find_pairs()

    print(
        f"Image/label pairs found: "
        f"{len(pairs)}"
    )

    if len(pairs) != 313:
        raise RuntimeError(
            f"Expected 313 images, "
            f"found {len(pairs)}"
        )

    # Safety: don't accidentally mix an old V2 dataset.
    if OUTPUT_ROOT.exists():

        print()
        print(
            f"Removing previous output:\n"
            f"{OUTPUT_ROOT}"
        )

        shutil.rmtree(
            OUTPUT_ROOT
        )

    train, val, test = split_pairs(
        pairs
    )

    print()
    print("Split sizes:")
    print(
        f"Train: {len(train)}"
    )
    print(
        f"Val:   {len(val)}"
    )
    print(
        f"Test:  {len(test)}"
    )

    print()
    print("Copying TRAIN...")
    train_counts = copy_split(
        "train",
        train
    )

    print("Copying VAL...")
    val_counts = copy_split(
        "val",
        val
    )

    print("Copying TEST...")
    test_counts = copy_split(
        "test",
        test
    )

    write_yaml()

    verify_output()

    print()
    print("=" * 70)
    print("INSTANCE COUNTS")
    print("=" * 70)

    for split_name, counts in [
        ("TRAIN", train_counts),
        ("VAL", val_counts),
        ("TEST", test_counts),
    ]:

        print()
        print(split_name)

        total = sum(
            counts.values()
        )

        print(
            f"  Total objects: {total}"
        )

        for class_id in [
            0, 1, 2
        ]:

            print(
                f"  "
                f"{CLASS_NAMES[class_id]:8}: "
                f"{counts[class_id]}"
            )

    print()
    print("=" * 70)
    print(
        "DATASET V2 CREATED SUCCESSFULLY"
    )
    print("=" * 70)

    print(
        f"\nOutput:\n{OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()