import os
import cv2
from flask import Flask, render_template
from model import resize_image, crop_center, grayscale_equalize, canny_edge, sobel_edge

app = Flask(__name__)

INPUT_DIR = "static/faces"
OUTPUT_DIR = "static/processed"

os.makedirs(INPUT_DIR, exist_ok=True)
subfolders = ["resized", "cropped", "grayscale_equalized", "canny", "sobel"]
for sf in subfolders:
    os.makedirs(os.path.join(OUTPUT_DIR, sf), exist_ok=True)


def process_images():
    for img_name in os.listdir(INPUT_DIR):
        img_path = os.path.join(INPUT_DIR, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        resized = resize_image(img)
        cropped = crop_center(resized)
        gray_eq = grayscale_equalize(resized)
        canny = canny_edge(resized)
        sobel = sobel_edge(resized)

        cv2.imwrite(os.path.join(OUTPUT_DIR, "resized", img_name), resized)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "cropped", img_name), cropped)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "grayscale_equalized", img_name), gray_eq)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "canny", img_name), canny)
        cv2.imwrite(os.path.join(OUTPUT_DIR, "sobel", img_name), sobel)


@app.route("/")
def index():
    images = []
    for img_name in os.listdir(INPUT_DIR):
        img_data = {
            "name": img_name,
            "original": f"faces/{img_name}",
            "resized": f"processed/resized/{img_name}",
            "cropped": f"processed/cropped/{img_name}",
            "gray_eq": f"processed/grayscale_equalized/{img_name}",
            "canny": f"processed/canny/{img_name}",
            "sobel": f"processed/sobel/{img_name}",
        }
        images.append(img_data)
    return render_template("index.html", images=images)


if __name__ == "__main__":
    process_images()
    print("Images processed. Starting Flask server...")
    app.run(debug=True)
