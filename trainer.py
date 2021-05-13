import os

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
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

custom_model_checkpoint_weights = os.path.join(checkpoints_dir, 'ckpted_custom_model_weights')

decoder_model_path = os.path.join(weights_dir, 'decoder_model')
decoder_model_weights = os.path.join(weights_dir, 'decoder_model_weights')


num_epochs = 5
batch_size = 4

caption_len = 25
images_per_story = 5
image_encoder_lstm_size = bi_lstm_size = 256


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

    text_encoder = text_encoders.get_embedding_layer(caption_len, words_to_idx)

    if os.path.isfile(vision_model_path) and os.path.isfile(decoder_model_path):
        print('Restored model that was explicitly saved!')
        vision_encoder = keras.models.load_model(vision_model_path)
        decoder_model = keras.models.load_model(decoder_model_path)
    else:
        print('Created new untrained model.')
        vision_encoder = vision_encoders.get_xception_encoder(images_per_story, image_encoder_lstm_size)
        decoder_model = decoders.get_decoder_bilstm(images_per_story, caption_len, image_encoder_lstm_size, bi_lstm_size, len(words_to_idx) + 1)

    return __make_model(text_encoder, vision_encoder, decoder_model, words_to_idx)


def predict_on_dev(model):
    dev_dataset = get_dataset(os.path.join(dev_tfrecords_dir, "dev-*.tfrecord"), batch_size)
    dev_batch = dev_dataset.take(2)

    expected_text_indices, got_text_softmax = model.predict(
        dev_batch,
        batch_size=2
    )

    got_text_softmax = tf.argmax(got_text_softmax, axis=-1).numpy()
    tokenizer = retrieve_tokenizer()

    for story in got_text_softmax:
        print('======== Predictions story ========')
        counter = 0
        for image in story:
            caption = tokenizer.sequences_to_texts(image)
            print('Predicted captions for image ' + str(counter) + ': ' + caption)
            counter += 1


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
        monitor="val_loss", factor=0.2, patience=1
    )

    # Create an early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )

    history = model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=dev_dataset,
        callbacks=[reduce_lr, early_stopping]
    )

    # Save model
    model.decoder.save_weights(decoder_model_weights, save_format='tf')
    model.vision_encoder.save_weights(vision_model_weights, save_format='tf')

    model.decoder.save(decoder_model_path)
    model.vision_encoder.save(vision_model_path)

    # Plot loss
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["train", "valid"], loc="upper right")
    plt.show()


keras.backend.clear_session()
train_model()
