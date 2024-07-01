import cv2
import numpy as np
from PIL import Image, ImageDraw

from yolov10.ultralytics import YOLOv10

CONF_THRESHOLD = 0.5
IMG_SIZE = 640

# Tên các lớp (classes) cho YOLOv10
CLASS_NAMES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "TV",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def load_model(model_path: str) -> YOLOv10:
    """
    Load the YOLOv10 model from the specified path.

    Args:
        model_path (str): Path to the YOLOv10 model file.

    Returns:
        YOLOv10: Loaded YOLOv10 model.
    """
    return YOLOv10(model_path)


def get_prediction(model: YOLOv10, image: np.ndarray) -> list:
    """
    Get predictions from the YOLOv10 model for the given image.

    Args:
        model (YOLOv10): Loaded YOLOv10 model.
        image (np.ndarray): Image in the form of a numpy array.

    Returns:
        list: List of prediction results from the model.
    """
    results = model.predict(source=image, imgsz=IMG_SIZE, conf=CONF_THRESHOLD)
    return results


def draw_predictions(
    image: np.ndarray,
    results: list,
    box_color: tuple = (0, 255, 0),
    label_color: tuple = (255, 255, 255),
    to_modify_class_name=True,
) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image based on the prediction results.

    Args:
        image (np.ndarray): Image in the form of a numpy array.
        results (list): List of prediction results from the model.
        box_color (tuple, optional): Color of the bounding box in BGR.
            Defaults to (0, 255, 0).
        label_color (tuple, optional): Color of the label text. Defaults to (255, 255, 255).

    Returns:
        np.ndarray: Image with bounding boxes and labels drawn.
    """
    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            conf = bbox.conf[0]
            class_id = int(bbox.cls[0])

            if not to_modify_class_name:
                label = CLASS_NAMES[class_id]
            else:
                label = ["head", "helmet", "person"][class_id]

            # Vẽ bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

            # Vẽ label và confidence
            text = f"{label}: {conf:.2f}"
            (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), box_color, -1)
            cv2.putText(
                image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2
            )
    return image


def draw_bounding_box(image_path, label_path):
    """Function to draw bounding boxes on an image"""
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    with open(label_path, "r") as file:
        for line in file:
            # Assuming the label file format is: class_id x_center y_center width height
            parts = line.strip().split()
            class_id, x_center, y_center, width, height = map(float, parts)

            # Convert normalized coordinates to absolute coordinates
            img_width, img_height = image.size
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height

            # Calculate the bounding box coordinates
            left = x_center - (width / 2)
            top = y_center - (height / 2)
            right = x_center + (width / 2)
            bottom = y_center + (height / 2)

            # Draw the bounding box on the image
            draw.rectangle([left, top, right, bottom], outline="red", width=2)

    return image
