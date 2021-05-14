import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.metrics import Mean

from tensorflow.keras.layers import Concatenate, Reshape
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

photos_per_story = 5


class CustomModel(Model):
    def __init__(self, vision_encoder, text_encoder, decoder, words_to_idx, words_per_caption, **kwargs):
        super(CustomModel, self).__init__(**kwargs)
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder
        self.decoder = decoder

        self.reshaper_image = Reshape(target_shape=(1, 299, 299, 3))
        self.reshaper_caption = Reshape(target_shape=(1, words_per_caption))
        self.sequence_concatenator = Concatenate(axis=1)
        self.text_vectorizator = TextVectorization(
            max_tokens=None,
            standardize=None,                           # Input is already standardized
            output_mode='int',
            output_sequence_length=words_per_caption,
            vocabulary=list(words_to_idx.keys())[1:]    # Exclude <OOV> token as it is automatically added
        )

        self.loss_tracker = Mean(name="loss")

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, training=False, **kwargs):
        correct_labels_tensors_to_concat = []
        images_to_concat = []

        for idx in range(photos_per_story):
            # Reshape image to (None, 1, 299, 299, 3) to concat later
            reshaped_image = self.reshaper_image(features['image_' + str(idx)])
            images_to_concat.append(reshaped_image)

            # Vectorize correct image caption => (None, words_per_caption)
            photo_correct_caption = self.text_vectorizator(features['caption_' + str(idx)])

            # Reshape correct image caption to (None, 1, words_per_caption)
            photo_correct_caption = self.reshaper_caption(photo_correct_caption)
            correct_labels_tensors_to_concat.append(photo_correct_caption)

        # Concat the correct captions of the whole story to form tensor of shape (None, 5, words_per_caption)
        expected_text_indices = self.sequence_concatenator(correct_labels_tensors_to_concat)

        # Concat images and get a tensor of size (None, 5, 299, 299, 3)
        photo_sequence = self.sequence_concatenator(images_to_concat)

        # Encode images => (None, 5, image_encoder_lstm_size)
        encoded_image_sequence = self.vision_encoder(photo_sequence)

        # Apply decoder model
        got_text_softmax = self.decoder(encoded_image_sequence)

        # (None, 5, words_per_caption), (None, 5, words_per_caption, vocab_len)
        return expected_text_indices, got_text_softmax


    def compute_loss(self, expected_text, predicted_text):
        embedding_distance_weight = 0.25
        # Compute categorical cross entropy between predicted captions and valid captions
        # Use sparse as we use the index words, not one-hot encodings.
        # use_logits=False as the predicted text represents the softmax activations
        hard_categorical_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=expected_text, y_pred=predicted_text
        )
        hard_categorical_loss = tf.keras.backend.batch_flatten(hard_categorical_loss)

        # y_pred and y_true of shapes (None, 5, words_per_caption)
        expected_text_indices = tf.keras.backend.cast(expected_text, tf.float32)
        predicted_text_indices = tf.keras.backend.cast(tf.keras.backend.argmax(predicted_text, axis=-1), tf.float32)

        # Compute the Manhattan distance between text embeddings
        expected_embeddings = self.text_encoder(expected_text_indices)
        predicted_embeddings = self.text_encoder(predicted_text_indices)

        embeddings_distance = tf.keras.backend.sum(tf.keras.backend.abs(expected_embeddings - predicted_embeddings),
                                                   axis=-1)

        # Return the mean of the loss over the batch
        return (hard_categorical_loss + embedding_distance_weight * embeddings_distance) / 2

    def train_step(self, features):
        with tf.GradientTape() as tape:
            # Forward pass
            expected_text, predicted_text = self(features, training=True)
            loss = self.compute_loss(expected_text, predicted_text)

        # Backward pass
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Monitor loss
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, features):
        expected_text, predicted_text = self(features, training=False)
        loss = self.compute_loss(expected_text, predicted_text)
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}
