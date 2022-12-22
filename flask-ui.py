from math import ceil, floor
from tkinter import Y
from flask import Flask, render_template, request
import os
from platform import python_branch

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from tensorflow.keras.preprocessing import image
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    jaccard_score,
    precision_score,
    recall_score,
)
from metrics import dice_loss, dice_coef, iou

H = 512
W = 512


def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_image(path):
    img = image.load_img(path, target_size=(299, 299))
    img = image.img_to_array(img) / 255
    img = np.array([img])
    return img


def read_mask(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
    ori_x = x
    x = x / 255.0
    x = x.astype(np.int32)
    return ori_x, x


def load_data(path):
    x = sorted(glob(os.path.join(path, "image", "*.jpg")))
    return x


def save_results(ori_x, ori_y, y_pred, save_image_path):
    line = np.ones((512, 10, 3)) * 255
    ori_y = np.expand_dims(ori_y, axis=-1)
    ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)
    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    cat_images = np.concatenate([ori_x, line, ori_y, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("test_page.html")


@app.route("/about")
def about_page():
    return "Image Segmentation"


@app.route("/submit", methods=["GET", "POST"])
def get_output():
    if request.method == "POST":
        img = request.files["image"]

        img_path = f"static\\{img.filename}.png"

        img.save(img_path)

        save_image_path = ""
        create_dir("flask_results")

        """ Load the model """
        with CustomObjectScope(
            {"iou": iou, "dice_coef": dice_coef, "dice_loss": dice_loss}
        ):
            model = tf.keras.models.load_model("files\\model.h5")

        """ Load the dataset """
        test_x = img_path
        print(len(test_x))

        """ Make the prediction and calculate the metrics values """
        SCORE = []
        print(test_x)
        if 1 == 1:
            x = test_x
            """ Extracting name """
            name = x.split("/")[-1].split(".")[0]

            print(x)
            """ Read the image and mask """
            x = read_image(x)

            """ Prediction """
            print(12)
            predictions = model.predict(x)
            print(predictions)
            predictions = (predictions > 0.5).astype("int32")
            recall_val = predictions
            if predictions[0][0] == 1:
                res = "SOME COVID SYMPTOMS ARE DETECTED IN PATIENTS XRAY"
            elif predictions[0][2] == 1:
                res = "SOME PNEUMONIA SYMPTOMS ARE DETECTED IN PATIENTS XRAY"
            else:
                res = "NORMAL"

            """ Saving the images """
            print("saved")
            print(name)

    return render_template(
        "test_page.html",
        acc_value=res,
        f1_value="0",
        jac_value="0",
        recall_value=str(recall_val),
        precision_value=test_x,
        img_path=test_x,
    )


if __name__ == "__main__":
    # app.debug = True
    app.run(debug=True)
