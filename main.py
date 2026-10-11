"""Face anti-spoofing detection using MiniFASNet models."""

import argparse
import warnings

import cv2
import numpy as np
import torch
from uniface import RetinaFace

from models import MiniFASNetV1SE, MiniFASNetV2
from utils import crop_face, draw_bbox, to_tensor, xyxy2xywh

warnings.filterwarnings("ignore")

MODEL_CONFIGS = {
    "v1se": {
        "class": MiniFASNetV1SE,
        "input_size": (80, 80),
        "scale": 4.0
    },
    "v2": {
        "class": MiniFASNetV2,
        "input_size": (80, 80),
        "scale": 2.7
    },
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Face Anti-Spoofing Detection"
    )

    parser.add_argument(
        "--weight",
        type=str,
        required=True,
        help="Path to model weights (.pth)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="v1se",
        choices=["v1se", "v2"],
        help="Model variant"
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Image path or camera index"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save output"
    )

    parser.add_argument(
        "--view",
        action="store_true",
        help="Display inference results"
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Face detection confidence threshold"
    )

    return parser.parse_args()


def load_model(weight_path, model_name, device):
    """Load anti-spoofing model."""

    config = MODEL_CONFIGS[model_name]

    model = config["class"]()

    model.load_state_dict(
        torch.load(
            weight_path,
            map_location=device,
            weights_only=True
        )
    )

    model.to(device).eval()

    print(f"Loaded {model_name} model on {device}")

    return model, config


def predict(image, bbox, model, config, device):
    """Predict whether a face is real or fake."""

    bbox_xywh = xyxy2xywh(bbox).astype(int).tolist()

    h, w = config["input_size"]

    face_crop = crop_face(
        image,
        bbox_xywh,
        config["scale"],
        w,
        h
    )

    tensor = to_tensor(face_crop).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()

    label_idx = int(np.argmax(probs))
    score = float(probs[0, label_idx])

    return {
        "label": "Real" if label_idx == 1 else "Fake",
        "score": score,
        "bbox": bbox_xywh,
    }


def run_image(args, model, config, detector, device):
    """Process an image only if exactly one face is detected."""

    image = cv2.imread(args.source)

    if image is None:
        raise ValueError(f"Failed to load image: {args.source}")

    faces = detector.detect(image)

    valid_faces = [
        face for face in faces
        if face.confidence >= args.confidence
    ]

    print("Faces detected:", len(valid_faces))

    if len(valid_faces) == 0:

        print("No face detected. Please show your face.")

    elif len(valid_faces) > 1:

        print("WARNING: Multiple faces detected!")
        print("Only one person is allowed.")
        print("Image processing blocked.")

    else:

        print("Exactly one face detected.")

        face = valid_faces[0]

        result = predict(
            image,
            face.bbox,
            model,
            config,
            device
        )

        draw_bbox(
            image,
            result["bbox"],
            result["label"],
            result["score"]
        )

        print(
            f"Face: {result['label']} "
            f"(score={result['score']:.2f})"
        )

    if args.output:
        cv2.imwrite(args.output, image)
        print(f"Saved: {args.output}")

    if args.view:
        cv2.imshow("Anti-Spoofing", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_webcam(args, model, config, detector, device):
    """Allow prediction only when exactly one face is detected."""

    camera_index = int(args.source) if args.source.isdigit() else 0

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise ValueError("Failed to open camera")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    writer = None

    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        writer = cv2.VideoWriter(
            args.output,
            fourcc,
            fps,
            (width, height)
        )

    print("Webcam started. Press 'q' to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read camera frame.")
            break

        # Detect faces in the current frame
        faces = detector.detect(frame)

        # Show raw detection count in terminal
        print("\nRaw faces detected:", len(faces))

        # Display confidence of each detected face
        for i, face in enumerate(faces):

            print(
                f"Face {i + 1} confidence: "
                f"{face.confidence:.3f}"
            )

        # Filter detections using confidence threshold
        valid_faces = [
            face for face in faces
            if face.confidence >= args.confidence
        ]

        face_count = len(valid_faces)

        print("Valid faces:", face_count)

        # Display detected face count
        cv2.putText(
            frame,
            f"Faces detected: {face_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        # CASE 1: No face detected
        if face_count == 0:

            cv2.putText(
                frame,
                "NO FACE DETECTED",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            print("No face detected.")

        # CASE 2: More than one face detected
        elif face_count > 1:

            cv2.putText(
                frame,
                "WARNING: MULTIPLE FACES!",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "ONLY ONE PERSON ALLOWED",
                (20, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

            print(
                "WARNING: Multiple faces detected! "
                "MiniFASNet prediction blocked."
            )

            # Do not process any face when multiple faces are detected

        # CASE 3: Exactly one face detected
        elif face_count == 1:

            face = valid_faces[0]

            result = predict(
                frame,
                face.bbox,
                model,
                config,
                device
            )

            draw_bbox(
                frame,
                result["bbox"],
                result["label"],
                result["score"]
            )

            print(
                f"Prediction: {result['label']} "
                f"(score={result['score']:.2f})"
            )

        # Save video if output is provided
        if writer is not None:
            writer.write(frame)

        # Display webcam window
        if args.view:

            cv2.imshow("Anti-Spoofing", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()

    if writer is not None:
        writer.release()
        print(f"Saved: {args.output}")

    cv2.destroyAllWindows()


def main():

    args = parse_args()

    if not args.view and not args.output:
        raise ValueError(
            "At least one of --view or --output must be provided"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, config = load_model(
        args.weight,
        args.model,
        device
    )

    detector = RetinaFace()

    # Determine whether the source is an image or webcam
    is_image = args.source.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )

    if is_image:

        run_image(
            args,
            model,
            config,
            detector,
            device
        )

    else:

        run_webcam(
            args,
            model,
            config,
            detector,
            device
        )


if __name__ == "__main__":
    main()