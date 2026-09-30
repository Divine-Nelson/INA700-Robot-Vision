from ultralytics import YOLO


def main():

    # Start from pretrained YOLO weights.
    model = YOLO("yolo11n.pt")

    results = model.train(
        data="dataset/data.yaml",

        epochs=100,

        imgsz=640,

        batch=16,

        patience=20,

        project="runs",

        name="abb_shapes",

        plots=True
    )

    print("Training complete.")

    print(
        "Best model:",
        results.save_dir / "weights" / "best.pt"
    )


if __name__ == "__main__":
    main()