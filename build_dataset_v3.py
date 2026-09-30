from pathlib import Path
from collections import Counter
import random
import shutil


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(r"H:\ABB_Vision_Project")

# ------------------------------------------------------------
# ORIGINAL V2 SOURCE
# ------------------------------------------------------------

V2_IMAGE_ROOT = PROJECT_ROOT / "basler_raw"
V2_LABEL_ROOT = PROJECT_ROOT / "basler_multi_labels"

# ------------------------------------------------------------
# NEW TARGETED V3 SOURCE
# ------------------------------------------------------------

EXTRA_IMAGE_ROOT = PROJECT_ROOT / "basler_multi_rawimages"
EXTRA_LABEL_ROOT = PROJECT_ROOT / "basler_combine_labels"

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_ROOT = PROJECT_ROOT / "dataset_3class_v3"


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


TRAIN_RATIO = 0.85

RANDOM_SEED = 42


# ============================================================
# HELPERS
# ============================================================

def find_pairs(image_root, label_root, source_name):
    """
    Find image/label pairs recursively.

    Empty label files are allowed because they represent
    intentional negative/background images.
    """

    pairs = []

    if not image_root.exists():
        raise FileNotFoundError(
            f"Image root not found:\n{image_root}"
        )

    if not label_root.exists():
        raise FileNotFoundError(
            f"Label root not found:\n{label_root}"
        )

    for image_path in sorted(image_root.rglob("*")):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        relative = image_path.relative_to(image_root)

        label_path = (
            label_root / relative
        ).with_suffix(".txt")

        if not label_path.exists():
            raise FileNotFoundError(
                f"Missing label for:\n"
                f"{image_path}\n\n"
                f"Expected:\n"
                f"{label_path}"
            )

        pairs.append(
            {
                "image": image_path,
                "label": label_path,
                "source": source_name,
                "relative": relative,
            }
        )

    return pairs


def split_pairs(pairs, train_ratio, seed):
    """
    Shuffle and split one source collection.

    Splitting the V2 and V3-extra collections separately
    guarantees that both collections contribute to train
    and validation.
    """

    items = pairs.copy()

    rng = random.Random(seed)
    rng.shuffle(items)

    train_count = round(
        len(items) * train_ratio
    )

    # Prevent pathological tiny-set splits.
    if len(items) > 1:
        train_count = max(
            1,
            min(train_count, len(items) - 1)
        )

    train = items[:train_count]
    val = items[train_count:]

    return train, val


def read_label_counts(label_path):
    """
    Count valid class IDs in one label.

    Empty labels simply return an empty Counter.
    """

    counts = Counter()

    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return counts

    for line in text.splitlines():

        parts = line.strip().split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid YOLO label line in:\n"
                f"{label_path}\n"
                f"{line}"
            )

        class_id = int(parts[0])

        if class_id not in CLASS_NAMES:
            raise ValueError(
                f"Invalid class ID {class_id} in:\n"
                f"{label_path}"
            )

        counts[class_id] += 1

    return counts


def make_output_directories():
    """
    Create a fresh V3 output directory.
    """

    if OUTPUT_ROOT.exists():

        print()
        print(
            f"Removing existing V3 dataset:\n"
            f"{OUTPUT_ROOT}"
        )

        shutil.rmtree(OUTPUT_ROOT)

    for split in ["train", "val"]:

        (
            OUTPUT_ROOT /
            "images" /
            split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            OUTPUT_ROOT /
            "labels" /
            split
        ).mkdir(
            parents=True,
            exist_ok=True,
        )


def destination_stem(item):
    """
    Generate collision-safe names.

    Examples:

    V2:
        v2_circle_raw_circle_0001

    New targeted:
        extra_combine_0001
    """

    relative = item["relative"]

    relative_without_suffix = (
        relative.with_suffix("")
    )

    parts = list(
        relative_without_suffix.parts
    )

    safe_relative = "_".join(parts)

    return (
        f"{item['source']}_"
        f"{safe_relative}"
    )


def copy_split(items, split_name):
    """
    Copy image/label pairs into the V3 dataset.
    """

    image_output = (
        OUTPUT_ROOT /
        "images" /
        split_name
    )

    label_output = (
        OUTPUT_ROOT /
        "labels" /
        split_name
    )

    for item in items:

        image_path = item["image"]
        label_path = item["label"]

        stem = destination_stem(item)

        destination_image = (
            image_output /
            f"{stem}{image_path.suffix.lower()}"
        )

        destination_label = (
            label_output /
            f"{stem}.txt"
        )

        shutil.copy2(
            image_path,
            destination_image,
        )

        # copy2 correctly preserves empty negative labels.
        shutil.copy2(
            label_path,
            destination_label,
        )


def summarize(items, title):
    """
    Print image count, target counts and negative-image count.
    """

    class_counts = Counter()

    negative_count = 0

    source_counts = Counter()

    for item in items:

        source_counts[
            item["source"]
        ] += 1

        counts = read_label_counts(
            item["label"]
        )

        if sum(counts.values()) == 0:
            negative_count += 1

        class_counts.update(counts)

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)

    print(
        f"Images:              {len(items)}"
    )

    print(
        f"Square instances:    {class_counts[0]}"
    )

    print(
        f"Circle instances:    {class_counts[1]}"
    )

    print(
        f"Triangle instances:  {class_counts[2]}"
    )

    print(
        f"Total targets:       {sum(class_counts.values())}"
    )

    print(
        f"Negative images:     {negative_count}"
    )

    print()

    print("Source contribution:")

    for source, count in sorted(
        source_counts.items()
    ):
        print(
            f"  {source}: {count}"
        )


def write_yaml():
    """
    Write YOLO dataset configuration.

    Relative path works when training from the dataset folder,
    but we'll replace it with the local C: absolute path after
    copying the dataset to the training computer.
    """

    yaml_text = """path: .
train: images/train
val: images/val

names:
  0: square
  1: circle
  2: triangle
"""

    yaml_path = (
        OUTPUT_ROOT /
        "data.yaml"
    )

    yaml_path.write_text(
        yaml_text,
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("BUILD INA700 V3 DATASET")
    print("=" * 70)

    # --------------------------------------------------------
    # Discover both source collections
    # --------------------------------------------------------

    v2_pairs = find_pairs(
        V2_IMAGE_ROOT,
        V2_LABEL_ROOT,
        "v2",
    )

    extra_pairs = find_pairs(
        EXTRA_IMAGE_ROOT,
        EXTRA_LABEL_ROOT,
        "extra",
    )

    print()
    print(
        f"Original V2 images:  {len(v2_pairs)}"
    )

    print(
        f"New targeted images: {len(extra_pairs)}"
    )

    print(
        f"Combined images:     "
        f"{len(v2_pairs) + len(extra_pairs)}"
    )

    # --------------------------------------------------------
    # Safety checks
    # --------------------------------------------------------

    if len(v2_pairs) != 313:

        print()
        print(
            "WARNING:"
        )

        print(
            f"Expected 313 original V2 images "
            f"but found {len(v2_pairs)}."
        )

        print(
            "The script will NOT continue because "
            "the frozen V2 source appears to have changed."
        )

        return

    if len(extra_pairs) != 50:

        print()
        print(
            "WARNING:"
        )

        print(
            f"Expected 50 targeted images "
            f"but found {len(extra_pairs)}."
        )

        print(
            "The script will NOT continue because "
            "the targeted source does not match the "
            "verified batch."
        )

        return

    # --------------------------------------------------------
    # Split each source independently.
    # --------------------------------------------------------

    v2_train, v2_val = split_pairs(
        v2_pairs,
        TRAIN_RATIO,
        RANDOM_SEED,
    )

    extra_train, extra_val = split_pairs(
        extra_pairs,
        TRAIN_RATIO,
        RANDOM_SEED + 1,
    )

    # --------------------------------------------------------
    # Combine the corresponding splits.
    # --------------------------------------------------------

    train_items = (
        v2_train +
        extra_train
    )

    val_items = (
        v2_val +
        extra_val
    )

    # Shuffle combined ordering only.
    # Membership has already been determined.
    rng = random.Random(
        RANDOM_SEED + 2
    )

    rng.shuffle(train_items)
    rng.shuffle(val_items)

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    all_destination_names = [
        destination_stem(item)
        for item in train_items + val_items
    ]

    if (
        len(all_destination_names)
        != len(set(all_destination_names))
    ):
        raise RuntimeError(
            "Destination filename collision detected."
        )

    if (
        len(train_items) +
        len(val_items)
        != 363
    ):
        raise RuntimeError(
            "Combined split does not contain "
            "all 363 expected images."
        )

    # --------------------------------------------------------
    # Create clean output
    # --------------------------------------------------------

    make_output_directories()

    # --------------------------------------------------------
    # Copy files
    # --------------------------------------------------------

    copy_split(
        train_items,
        "train",
    )

    copy_split(
        val_items,
        "val",
    )

    # --------------------------------------------------------
    # Write YAML
    # --------------------------------------------------------

    write_yaml()

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    summarize(
        train_items,
        "TRAIN SPLIT",
    )

    summarize(
        val_items,
        "VALIDATION SPLIT",
    )

    summarize(
        train_items + val_items,
        "FULL V3 DATASET",
    )

    print()
    print("=" * 70)
    print("V3 DATASET BUILD COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Dataset created at:\n"
        f"{OUTPUT_ROOT}"
    )

    print()

    print(
        "IMPORTANT:"
    )

    print(
        "This V3 dataset is for model development."
    )

    print(
        "A fresh Basler dataset should be collected "
        "after the V3 model is frozen for the final "
        "independent evaluation."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()