from ultralytics import YOLO
import argparse


def main(args):

    # Load base YOLOv8n pretrained weights
    model = YOLO("yolov8n.pt")

    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        optimizer="AdamW",
        lr0=0.001,
        patience=10,
        cls=1.2,
        box=7.5,
        dfl=1.5,
        mosaic=0.2,
        mixup=0.0,
        project="runs/train",
        name="orings_yolov8n",
        exist_ok=True
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)

    args = parser.parse_args()
    main(args)
