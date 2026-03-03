import cv2
import time
import argparse
from ultralytics import YOLO


def run_inference(weights_path, source, conf_threshold):

    # Load trained YOLO model
    try:
        model = YOLO(weights_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Initialize camera
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    print("Real-time inference started. Press 'q' to exit.")

    prev_time = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed.")
            break

        # Run inference
        results = model(frame, conf=conf_threshold, verbose=False)[0]

        # Annotated frame
        annotated_frame = results.plot()

        # FPS calculation
        current_time = time.time()
        fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
        prev_time = current_time

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.2f}",
            (15, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("O-Ring Defect Detection", annotated_frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="YOLOv8 Real-Time O-Ring Defect Detection"
    )

    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights (best.pt)"
    )

    parser.add_argument(
        "--source",
        type=int,
        default=0,
        help="Camera source index (default: 0)"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )

    args = parser.parse_args()

    run_inference(args.weights, args.source, args.conf)
