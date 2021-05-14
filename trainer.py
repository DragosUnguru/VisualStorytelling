import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import data_prep
import decoders
import text_encoders
import vision_encoders

os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import custom_model
import tensorflow as tf
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from data_prep import get_dataset
from vocabulary_builder import retrieve_tokenizer

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

train_tfrecords_dir = os.path.join(os.getcwd(), 'data\\VIST\\train\\tfrecords')
dev_tfrecords_dir = os.path.join(os.getcwd(), 'data\\VIST\\dev\\tfrecords')

weights_dir = 'C:\\Users\\ungur\\Desktop\\licentaV2\\weights'
checkpoints_dir = os.path.join(weights_dir, 'checkpoints')

vision_model_path = os.path.join(weights_dir, 'vision_model')
vision_model_weights = os.path.join(weights_dir, 'vision_model_weights')

text_model_path = os.path.join(weights_dir, 'text_model')
text_model_weights = os.path.join(weights_dir, 'text_model_weights')

custom_model_checkpoint_weights = os.path.join(checkpoints_dir, 'ckpted_custom_model_weights')

decoder_model_path = os.path.join(weights_dir, 'decoder_model')
decoder_model_weights = os.path.join(weights_dir, 'decoder_model_weights')


num_epochs = 30
batch_size = 2

caption_len = 25
images_per_story = 5
image_encoder_lstm_size = text_encoder_lstm_size = bi_lstm_size = 1024


def __make_model(text_encoder, vision_encoder, decoder, words_to_idx):
    my_custom_model = custom_model.CustomModel(
        vision_encoder=vision_encoder,
        text_encoder=text_encoder,
        decoder=decoder,
        words_to_idx=words_to_idx,
        words_per_caption=caption_len,
    )
    my_custom_model.compile(
        optimizer=keras.optimizers.Adam()
    )

    return my_custom_model


def make_or_restore_model():
    tokenizer = retrieve_tokenizer()
    words_to_idx = tokenizer.word_index

    if os.path.isdir(vision_model_path) and os.path.isdir(decoder_model_path) and os.path.isdir(text_model_path):
        text_encoder = keras.models.load_model(text_model_path)
        vision_encoder = keras.models.load_model(vision_model_path)
        decoder_model = keras.models.load_model(decoder_model_path)
        print('Restored model!')
    else:
        print('Creating new untrained model...')
        text_encoder = text_encoders.glove_embedding_encoder(caption_len, words_to_idx, text_encoder_lstm_size)
        vision_encoder = vision_encoders.get_xception_encoder(images_per_story, image_encoder_lstm_size)
        decoder_model = decoders.get_decoder_bilstm(images_per_story, caption_len, image_encoder_lstm_size, bi_lstm_size, len(words_to_idx) + 1)

    return __make_model(text_encoder, vision_encoder, decoder_model, words_to_idx)


def predict_on_dev():
    model = make_or_restore_model()

    dev_dataset = get_dataset(os.path.join(dev_tfrecords_dir, "dev-*.tfrecord"), batch_size)
    batch = next(iter(dev_dataset))

    expected_text_indices, got_text_softmax = model.predict(batch)

    got_text_softmax = tf.argmax(got_text_softmax, axis=-1).numpy()
    tokenizer = retrieve_tokenizer()

    print('======== Predictions story ========')
    for story in got_text_softmax:
        print('PREDICTED CAPTIONS FOR NEW STORY\n')
        caption = tokenizer.sequences_to_texts(story.tolist())
        print(caption)

    print('\n\n======== Correct story ========')
    for story in expected_text_indices:
        print('CORRECT CAPTIONS FOR NEW STORY\n')
        caption = tokenizer.sequences_to_texts(story.tolist())
        print(caption)


def evaluate_on_dev(model):
    dev_dataset = get_dataset(os.path.join(dev_tfrecords_dir, "dev-*.tfrecord"), batch_size)

    loss, acc = model.evaluate(dev_dataset, verbose=2)
    print("Untrained model, accuracy: {:5.2f}%".format(100 * acc))


def train_model():
    model = make_or_restore_model()
    train_dataset = get_dataset(os.path.join(train_tfrecords_dir, "train-*.tfrecord"), batch_size)
    dev_dataset = get_dataset(os.path.join(dev_tfrecords_dir, "dev-*.tfrecord"), batch_size)

    # Create a learning rate scheduler callback
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.2, patience=3
    )

    # Create an early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )

    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=dev_dataset,
        callbacks=[reduce_lr, early_stopping]
    )

    # Save model
    model.text_encoder.save_weights(text_model_weights, save_format='tf')
    model.decoder.save_weights(decoder_model_weights, save_format='tf')
    model.vision_encoder.save_weights(vision_model_weights, save_format='tf')

    model.text_encoder.save(text_model_path)
    model.decoder.save(decoder_model_path)
    model.vision_encoder.save(vision_model_path)

    # Plot loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "valid"], loc="upper right")
    plt.show()

    print(model.summary())


tf.keras.backend.clear_session()
train_model()
# predict_on_dev()
