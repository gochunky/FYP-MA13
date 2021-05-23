"""
Main module for the project. Run this file with commands like this:

python classify_gender.py {IMAGE} [-p]

{IMAGE} is the source image filepath and [-p] or [--perturbed] is used to
determine whether we test with the perturbed or unperturbed model. If -p is not
supplied, use the unperturbed (baseline) model.
"""

import argparse
import cv2
import tensorflow as tf


def parse_args():
    """
    Argument parser that accept arguments from the terminal.
    
    :return: argument namespace. Use this as the arguments to run the program.
    """
    parser = argparse.ArgumentParser(
        description="Perform gender classification on a single image"
    )
    parser.add_argument("image", help="input image")
    parser.add_argument(
        "-p",
        "--perturbed",
        help="if True, use the perturbed model to test",
        action="store_true"
    )
    return parser.parse_args()


def preprocess(img_path, img_size=224):
    """
    Preprocess a single image to be compatible with our models.
    
    :param img_path: filepath to image
    :param img_size: height and width of image
    :return: a single image matrix compatible with classification
    """
    # Convert images from BGR to RGB format
    img_arr = cv2.imread(img_path)
    # Reshaping the arrays to a form that can be processed
    resized_arr = cv2.resize(img_arr, (img_size, img_size))
    updated = resized_arr / 255
    updated.reshape(-1, img_size, img_size, 1)
    return updated


def load_model(is_perturbed):
    """
    Load a single model.

    :param is_perturbed: a boolean indicating whether the model has been trained
        with perturbed data
    :return: the corresponding model
    """
    model_path = f"model_best_weights{'_pert' if is_perturbed else ''}.h5"
    return tf.keras.models.load_model(model_path)


def classify(model, img):
    """
    Perform a binary classification on an image.

    :param model: model to perform classification
    :param img: image to classify
    :return: a tuple indicating (result, confidence) for the classification
    """
    return model.predict(img)[0], model.predict_proba(img)[0]


def main():
    """Driver code. Run everything here."""
    labels = ["Female", "Male"]
    args = parse_args()
    img = preprocess(args.image)
    model = load_model(args.perturbed)
    result, confidence = classify(model, img)
    print(f"Classification for {img}: {labels[result]}")
    print(f"Confidence: {round(confidence * 100, 2)}%")


if __name__ == "__main__":
    main()
