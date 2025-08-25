import cv2
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

resize_transform = transforms.Resize((128, 128))
center_crop = transforms.CenterCrop(100)


def resize_image(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    resized = resize_transform(pil_img)
    return cv2.cvtColor(np.array(resized), cv2.COLOR_RGB2BGR)


def crop_center(img):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    cropped = center_crop(pil_img)
    return cv2.cvtColor(np.array(cropped), cv2.COLOR_RGB2BGR)


def grayscale_equalize(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    return equalized


def canny_edge(img):
    edges = cv2.Canny(img, 100, 200)
    return edges


def sobel_edge(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(grad_x, grad_y)
    sobel = np.uint8(sobel / np.max(sobel) * 255)
    return sobel


