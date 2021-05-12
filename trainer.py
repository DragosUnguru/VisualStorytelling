import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import custom_model
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import decoders
import text_encoders
import vision_encoders
from data_prep import get_dataset
from vocabulary_builder import retrieve_tokenizer

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

train_tfrecords_dir = os.path.join(os.getcwd(), 'data\\VIST\\train\\tfrecords')
dev_tfrecords_dir = os.path.join(os.getcwd(), 'data\\VIST\\dev\\tfrecords')
checkpoint_dir = 'C:\\Users\\ungur\\Desktop\\licentaV2\\weights'
best_model_path = os.path.join(checkpoint_dir, 'best_model')

num_epochs = 4
batch_size = 8

caption_len = 25

prev_sentence_encoder_LSTM_size = 256
image_encoder_output_size = 4096

pre_att_LSTM_size = 64
post_att_LSTM_size = 32

image_encoder_lstm_size = 256
bi_lstm_size = 256


def __make_model():
    tokenizer = retrieve_tokenizer()
    words_to_idx = tokenizer.word_index
    vocab_size = len(words_to_idx) + 1

    text_encoder = text_encoders.get_embedding_layer(caption_len, words_to_idx)
    vision_encoder = vision_encoders.get_xception_encoder()
    decoder = decoders.get_decoder_bilstm(
        photos_per_story=5,
        words_per_caption=caption_len,
        input_features_len=image_encoder_lstm_size,
        lstm_size=image_encoder_lstm_size,
        vocab_size=vocab_size
    )

    my_mare_model = custom_model.DecoderModel(
        text_encoder=text_encoder,
        vision_encoder=vision_encoder,
        decoder=decoder,
        words_to_idx=words_to_idx,
        idx_to_word=tokenizer.index_word,
        words_per_caption=caption_len,
        image_encoder_lstm_size=image_encoder_lstm_size
    )

    my_mare_model.compile(
        optimizer=tf.optimizers.Adam()
    )

    return my_mare_model


def make_or_restore_model():
    # Either restore the best model saved from early stopping callback,
    # either the latest model saved via checkpointing,
    # or create a fresh one if there is no checkpoint available.
    if os.path.isfile(best_model_path):
        print('Restored best model saved from EalyStopping callback')
        return keras.models.load_model(best_model_path)

    checkpoints = [os.path.join(checkpoint_dir, '\\' + name)
                   for name in os.listdir(checkpoint_dir)]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print('Restoring from', latest_checkpoint)
        return keras.models.load_model(latest_checkpoint)

    print('Creating a new model')
    return __make_model()


def train_model():
    model = make_or_restore_model()
    train_dataset = get_dataset(os.path.join(train_tfrecords_dir, "train-*.tfrecord"), batch_size)
    dev_dataset = get_dataset(os.path.join(dev_tfrecords_dir, "dev-*.tfrecord"), batch_size)

    # Create a learning rate scheduler callback
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=1
    )

    # Create an early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )

    # Create checkpointing callback
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(checkpoint_dir, '\\ckpted_model'),
        save_freq=500
    )

    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=dev_dataset,
        batch_size=batch_size,
        callbacks=[reduce_lr, early_stopping, checkpoint]
    )

    model.save_weights(os.path.join(checkpoint_dir, 'model_weights.h5'))
    model.decoder.save_weights(os.path.join(checkpoint_dir, 'decoder_model_weights.h5'))
    model.save(best_model_path)

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "valid"], loc="upper right")
    plt.show()


tf.keras.backend.clear_session()
train_model()
