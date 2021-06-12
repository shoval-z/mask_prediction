import os
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib import patches

np.random.seed(42)
image_dir = "example_images"


def parse_images_and_bboxes(image_dir):
    """
    Parse a directory with images.
    :param image_dir: Path to directory with images.
    :return: A list with (filename, image_id, bbox, proper_mask) for every image in the image_dir.
    """
    example_filenames = os.listdir(image_dir)
    data = []
    for filename in example_filenames:
        image_id, bbox, proper_mask = filename.strip(".jpg").split("__")
        bbox = json.loads(bbox)
        proper_mask = True if proper_mask.lower() == "true" else False
        data.append((filename, image_id, bbox, proper_mask))
    return data


def calc_iou(bbox_a, bbox_b):
    """
    Calculate intersection over union (IoU) between two bounding boxes with a (x, y, w, h) format.
    :param bbox_a: Bounding box A. 4-tuple/list.
    :param bbox_b: Bounding box B. 4-tuple/list.
    :return: Intersection over union (IoU) between bbox_a and bbox_b, between 0 and 1.
    """
    x1, y1, w1, h1 = bbox_a
    x2, y2, w2, h2 = bbox_b
    w_intersection = min(x1 + w1, x2 + w2) - max(x1, x2)
    h_intersection = min(y1 + h1, y2 + h2) - max(y1, y2)
    if w_intersection <= 0.0 or h_intersection <= 0.0:  # No overlap
        return 0.0
    intersection = w_intersection * h_intersection
    union = w1 * h1 + w2 * h2 - intersection    # Union = Total Area - Intersection
    return intersection / union


def show_images_and_bboxes(data, image_dir):
    """
    Plot images with bounding boxes. Predicts random bounding boxes and computes IoU.
    :param data: Iterable with (filename, image_id, bbox, proper_mask) structure.
    :param image_dir: Path to directory with images.
    :return: None
    """
    for filename, image_id, bbox, proper_mask in data:
        # Load image
        im = cv2.imread(os.path.join(image_dir, filename))
        # BGR to RGB
        im = im[:, :, ::-1]
        # Ground truth bbox
        x1, y1, w1, h1 = bbox
        # Predicted bbox
        x2, y2, w2, h2 = random_bbox_predict(bbox)
        # Calculate IoU
        iou = calc_iou(bbox, (x2, y2, w2, h2))
        # Plot image and bboxes
        fig, ax = plt.subplots()
        ax.imshow(im)
        rect = patches.Rectangle((x1, y1), w1, h1,
                                 linewidth=2, edgecolor='g', facecolor='none', label='ground-truth')
        ax.add_patch(rect)
        rect = patches.Rectangle((x2, y2), w2, h2,
                                 linewidth=2, edgecolor='b', facecolor='none', label='predicted')
        ax.add_patch(rect)
        fig.suptitle(f"proper_mask={proper_mask}, IoU={iou:.2f}")
        ax.axis('off')
        fig.legend()
        plt.show()


def random_bbox_predict(bbox):
    """
    Randomly predicts a bounding box given a ground truth bounding box.
    For example purposes only.
    :param bbox: Iterable with numbers.
    :return: Random bounding box, relative to the input bbox.
    """
    return [x + np.random.randint(-15, 15) for x in bbox]


if __name__ == "__main__":
    data = parse_images_and_bboxes(image_dir)
    show_images_and_bboxes(data, image_dir)
