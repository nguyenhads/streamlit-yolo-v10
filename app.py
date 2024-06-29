import os

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from utils import CLASS_NAMES
from yolov10.ultralytics import YOLOv10

# Đường dẫn đến mô hình và các tham số khác
MODEL_PATH = os.path.join(os.getcwd(), "model/yolov10n.pt")
CONF_THRESHOLD = 0.5
IMG_SIZE = 640


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
) -> np.ndarray:
    """
    Draw bounding boxes and labels on the image based on the prediction results.

    Args:
        image (np.ndarray): Image in the form of a numpy array.
        results (list): List of prediction results from the model.
        box_color (tuple, optional): Color of the bounding box in BGR. Defaults to (0, 255, 0).
        label_color (tuple, optional): Color of the label text. Defaults to (255, 255, 255).

    Returns:
        np.ndarray: Image with bounding boxes and labels drawn.
    """
    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            conf = bbox.conf[0]
            class_id = int(bbox.cls[0])
            label = CLASS_NAMES[class_id]

            # Vẽ bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)

            # Vẽ label và confidence
            text = f"{label}: {conf:.2f}"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), box_color, -1)
            cv2.putText(
                image, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2
            )
    return image


def main():
    """
    Main function to run the Streamlit app.
    """
    st.title("YOLOv10 Image Recognition")
    st.write("Upload your image")

    uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Chuyển đổi ảnh sang định dạng OpenCV
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # Tải mô hình
        model = load_model(MODEL_PATH)

        # Nhận dự đoán
        results = get_prediction(model, image_np)

        # Vẽ kết quả dự đoán lên ảnh
        image_with_predictions = draw_predictions(
            image_np.copy(), results, box_color=(0, 255, 0), label_color=(0, 0, 0)
        )

        # Chuyển đổi ảnh lại sang định dạng RGB để hiển thị trên Streamlit
        image_with_predictions = cv2.cvtColor(image_with_predictions, cv2.COLOR_BGR2RGB)

        st.image(
            image_with_predictions,
            caption="Image with YOLOv10 Predictions.",
            use_column_width=True,
        )


if __name__ == "__main__":
    main()
