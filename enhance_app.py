import os
import time
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import streamlit as st

os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"


def preprocess_image(image_path):
    """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """

    pil_img = Image.open(image_path)
    hr_image = tf.convert_to_tensor(np.array(pil_img))
    # If PNG, remove the alpha channel. The model only supports
    # images with 3 color channels.
    if hr_image.shape[-1] == 4:
        hr_image = hr_image[..., :-1]
    hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
    hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
    hr_image = tf.cast(hr_image, tf.float32)
    return tf.expand_dims(hr_image, 0)


def enhance(file_path):
    # start = time.time()
    hr_image = preprocess_image(file_path)
    model = hub.load("https://tfhub.dev/captain-pool/esrgan-tf2/1")

    fake_image = model(hr_image)
    fake_image = tf.squeeze(fake_image)
    # print("Time Taken: %f" % (time.time() - start))
    return fake_image


def main_loop():
    st.title("OpenCV Demo App")
    st.subheader("This app allows you to play with Image filters!")
    st.text("We use Tensorflow and Streamlit for this demo")

    image_file = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg', 'tiff'])
    if not image_file:
        return None

    original_image = Image.open(image_file)
    enhance_image = enhance(image_file)
    enhance_image = tf.clip_by_value(enhance_image, 0, 255)
    enhance_image = Image.fromarray(tf.cast(enhance_image, tf.uint8).numpy())

    st.text("Original Image vs Enhanced Image")
    st.image([original_image, enhance_image])


if __name__ == '__main__':
    main_loop()
