import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.metrics import Mean

from tensorflow.keras.layers import Concatenate, Reshape, LSTM, TimeDistributed, Masking
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

import decoders

photos_per_story = 5


class DecoderModel(Model):
    def __init__(self, text_encoder, vision_encoder, decoder, words_to_idx, idx_to_word, words_per_caption, image_encoder_lstm_size, **kwargs):
        super(DecoderModel, self).__init__(**kwargs)
        self.text_encoder = text_encoder
        self.text_embedder = TimeDistributed(text_encoder)
        self.vision_encoder = vision_encoder
        self.decoder = decoder

        self.words_to_idx = words_to_idx
        self.idx_to_word = idx_to_word
        self.words_per_caption = words_per_caption
        self.loss_tracker = Mean(name="loss")

        self.image_encoder_lstm_size = image_encoder_lstm_size
        self.vision_encoder_out_size = self.vision_encoder.layers[-1].output_shape[-1]

        self.masking = Masking(mask_value=0.0)
        self.reshaper = Reshape(target_shape=(1, self.vision_encoder_out_size))
        self.reshaper_labels = Reshape(target_shape=(1, words_per_caption))
        self.sequence_concatenator = Concatenate(axis=-2)
        self.sequence_concatenator_2 = Concatenate(axis=-2)
        self.image_sequence_LSTM = LSTM(image_encoder_lstm_size, return_sequences=True)
        self.text_vectorizator = TextVectorization(
            max_tokens=None,                            # Exclude <OOV> and the reserved 0 as the output_mode = 'int'
            standardize=None,                           # Input is already standardized
            output_mode='int',
            output_sequence_length=words_per_caption,
            vocabulary=list(self.words_to_idx.keys())[1:]    # Exclude <OOV> token as it is automatically added
        )

    @property
    def metrics(self):
        return [self.loss_tracker]

    def call(self, features, training=False):
        vision_tensors_to_concat = []
        correct_labels_tensors_to_concat = []

        for idx in range(photos_per_story):
            # Encode image => (None, vision_encoder_out_size)
            image_embedding = self.vision_encoder(features['image_' + str(idx)])

            # Reshape image visual features to (None, 1, vision_encoder_out_size)
            reshaped_image_embedding = self.reshaper(image_embedding)
            vision_tensors_to_concat.append(reshaped_image_embedding)

            # Vectorize correct image caption => (None, words_per_caption)
            photo_correct_caption = self.text_vectorizator(features['caption_' + str(idx)])

            # Reshape correct image caption to (None, 1, words_per_caption)
            photo_correct_caption = self.reshaper_labels(photo_correct_caption)
            correct_labels_tensors_to_concat.append(photo_correct_caption)

        # Concat the correct captions of the whole story to form tensor of shape (None, 5, words_per_caption)
        expected_text_indices = self.sequence_concatenator_2(correct_labels_tensors_to_concat)

        # Concat images and get a tensor of size (None, 5, vision_encoder_out_size)
        lstm_input = self.sequence_concatenator(vision_tensors_to_concat)
        lstm_input = self.masking(lstm_input)

        # Apply LSTM encoding, get a tensor of size (None, 5, image_encoder_lstm_size)
        lstm_output = self.image_sequence_LSTM(lstm_input)

        # Apply decoder model
        got_text_softmax = self.decoder(lstm_output)

        # (None, 5, words_per_caption), (None, 5, words_per_caption, vocab_len)
        return expected_text_indices, got_text_softmax

    def compute_loss(self, expected_text, predicted_text):
        # Compute categorical cross entropy between predicted captions and valid captions
        # Use sparse as we use the index words, not one-hot encodings.
        # use_logits=False as the predicted text represents the softmax activations
        expected_text = tf.cast(expected_text, tf.float64)
        hard_loss = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=expected_text, y_pred=predicted_text
        )

        # Compute captions similarity using GloVe300D embeddings
        predicted_text_indices = tf.cast(tf.argmax(predicted_text, axis=-1), tf.float64)

        expected_embeddings = self.text_embedder(expected_text)
        predicted_embeddings = self.text_embedder(predicted_text_indices)

        # Use cosine similarity with pre-trained GloVe embeddings
        cosine_sim = tf.keras.losses.CosineSimilarity()
        embedding_loss = cosine_sim(expected_embeddings, predicted_embeddings)

        # Return the mean of the loss over the batch
        return (hard_loss + (embedding_loss + 1)) / 2

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

    def get_config(self):
        cfg = super(DecoderModel, self).get_config()
        cfg.update({
            "words_to_idx": self.words_to_idx,
            "idx_to_word": self.idx_to_word,
            "words_per_caption": self.words_per_caption,
            "image_encoder_lstm_size": self.image_encoder_lstm_size,
            "text_encoder": self.text_encoder,
            "vision_encoder": self.vision_encoder,
            "decoder": self.decoder,
            "compute_loss": self.compute_loss
        })
        return cfg
