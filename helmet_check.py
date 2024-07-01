import argparse
import os
from pathlib import Path
from typing import List, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import draw_bounding_box, draw_predictions, get_prediction, load_model

# Finetuned model on Helmet Safety Detection Data
MODEL_PATH = os.path.join(os.getcwd(), "model/helmet_check_50epochs.pt")
CONF_THRESHOLD = 0.5
IMG_SIZE = 640


def visualize_train_images(
    image_files: List[Union[str, Path]],
    image_folder_path: Union[str, Path],
    label_folder_path: Union[str, Path],
    num_images_to_display: int = 5,
):
    """
    Visualize training images with bounding boxes.

    Args:
        image_files (List[Union[str, Path]]): List of image file names.
        image_folder_path (Union[str, Path]): Path to the folder containing images.
        label_folder_path (Union[str, Path]): Path to the folder containing labels.
        num_images_to_display (int, optional): Number of images to display. Defaults to 5.
    """
    plt.figure(figsize=(10, 10))

    for i in range(min(num_images_to_display, len(image_files))):
        image_file = image_files[i]
        image_path = os.path.join(image_folder_path, image_file)
        label_path = os.path.join(
            label_folder_path,
            image_file.replace(".jpg", ".txt")
            .replace(".jpeg", ".txt")
            .replace(".png", ".txt"),
        )

        if os.path.exists(label_path):
            image_with_boxes = draw_bounding_box(image_path, label_path)

            # Display the image with bounding boxes
            plt.subplot(1, num_images_to_display, i + 1)
            plt.imshow(image_with_boxes)
            plt.title(f"Image {i + 1}")
            plt.axis("off")
        else:
            print(f"Label file not found for {image_file}")

    plt.show()


def main(test_image_index: int, to_visualize_train_images=False):
    image_folder_path = Path("./safety_helmet_dataset/train/images")
    label_folder_path = Path("./safety_helmet_dataset/train/labels")
    test_image_folder_path = Path("./safety_helmet_dataset/test/images")

    image_files = [
        f
        for f in os.listdir(image_folder_path)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    test_image_files = [
        f
        for f in os.listdir(test_image_folder_path)
        if f.endswith((".png", ".jpg", ".jpeg"))
    ]

    # Check training images
    if to_visualize_train_images:
        visualize_train_images(
            image_files=image_files,
            num_images_to_display=5,
            image_folder_path=image_folder_path,
            label_folder_path=label_folder_path,
        )

    model = load_model(MODEL_PATH)
    print(model.info())

    test_image_path = test_image_folder_path / test_image_files[test_image_index]

    image = Image.open(test_image_path)
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = get_prediction(model, image_np)

    image_with_predictions = draw_predictions(
        image_np.copy(), results, box_color=(0, 255, 0), label_color=(0, 0, 0)
    )

    image_with_predictions = cv2.cvtColor(image_with_predictions, cv2.COLOR_BGR2RGB)
    plt.imshow(image_with_predictions)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helmet Safety Detection")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the test image to predict (default: 0)",
    )
    args = parser.parse_args()
    main(test_image_index=args.index)
