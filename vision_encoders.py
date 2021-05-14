import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, TimeDistributed, Reshape, Concatenate, LSTM
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input

import slicer_layer


def get_xception_encoder(images_per_story, image_encoder_lstm_size):
    # Load the pre-trained Xception model to be used as the image encoder.
    xception = Xception(
        include_top=False, weights="imagenet", pooling="avg"
    )

    # Don't train the xception model
    for layer in xception.layers:
        layer.trainable = False

    inputs = Input(shape=(images_per_story, 299, 299, 3), name="images_input")

    reshaper = Reshape(target_shape=(1, xception.layers[-1].output_shape[-1]))
    concatenator = Concatenate(axis=1)
    image_embeddings_to_concat = []

    # Receive the images as inputs and slice on time axis
    seq_input = slicer_layer.TimestepSliceLayer()(inputs)

    for photo in seq_input:
        # Apply xception model on each photo
        preprocessed_photo = preprocess_input(tf.cast(photo, tf.float32))
        xception_output = xception(preprocessed_photo)
        reshaped_xception = reshaper(xception_output)
        image_embeddings_to_concat.append(reshaped_xception)

    lstm_input = concatenator(image_embeddings_to_concat)

    # Apply LSTM encoding, get a output of size (None, images_per_story, image_encoder_lstm_size)
    outputs = LSTM(image_encoder_lstm_size, return_sequences=True)(lstm_input)

    # Create the vision encoder model
    return Model(inputs, outputs, name="vision_encoder")
