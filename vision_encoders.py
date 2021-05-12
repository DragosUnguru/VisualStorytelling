from tensorflow.keras import Model
from tensorflow.keras.layers import Input, LSTM
from tensorflow.keras.applications import Xception
from tensorflow.keras.applications.xception import preprocess_input


def get_xception_encoder():
    # Load the pre-trained Xception model to be used as the image encoder.
    xception = Xception(
        include_top=False, weights="imagenet", pooling="avg"
    )

    # Don't train the encoder
    for layer in xception.layers:
        layer.trainable = False

    # Receive the images as inputs
    inputs = Input(shape=(299, 299, 3), name="images_input")

    # Preprocess the input images
    xception_input = preprocess_input(inputs)

    # Generate the embeddings for the image using the xception model
    embeddings = xception(xception_input)

    # Create the vision encoder model
    return Model(inputs, embeddings, name="vision_encoder")
