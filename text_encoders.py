import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding, Reshape, LSTM, TimeDistributed


photos_per_story = 5


def get_embedding_layer(sentence_len, words_to_idx):
    embeddings_index = {}
    output_size = 300

    glove_embeddings_path = 'C:\\Users\\ungur\\Desktop\\licentaV2\\data\\VIST\\glove.6B.' + str(output_size) + 'd.txt'
    f = open(glove_embeddings_path, encoding='utf8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    # try:
    #     # If there is an 'unk' word embedding, use that
    #     embedding_matrix =
    # except IndexError:

    embedding_matrix = np.zeros((len(words_to_idx) + 1, output_size))

    for word, i in words_to_idx.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    embedding_layer = Embedding(len(words_to_idx) + 1,
                                output_size,
                                weights=[embedding_matrix],
                                mask_zero=True,
                                input_length=sentence_len,
                                trainable=False)
    return embedding_layer


def bert_text_encoder():
    # Load the BERT preprocessing module.
    preprocess = hub.KerasLayer(
        'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3',
        name='text_preprocessing'
    )

    # Load the pre-trained BERT model to be used as the base encoder.
    bert = hub.KerasLayer(
        'https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/2',
        trainable=False,
        name='bert'
    )

    # We don't want to train the encoder model
    bert.trainable = False

    # Receive the text as inputs
    inputs = Input(shape=(), dtype=tf.string, name="text_input")

    # Preprocess the text
    bert_inputs = preprocess(inputs)

    # Generate embeddings for the preprocessed text using the BERT model
    embeddings = bert(bert_inputs)["pooled_output"]

    # Project the embeddings produced by the model
    # outputs = project_embeddings(embeddings, projection_dims, dropout_rate)

    # Create the text encoder model
    return Model(inputs, embeddings, name="text_encoder")


def manhattan_distance(A, B):
    return tf.keras.sum(tf.keras.abs(A - B), axis=1, keepdims=True)


def glove_embedding_encoder(words_per_caption, words_to_idx, lstm_size):
    # Receive inputs of shape (None, 5, words_per_caption)
    inputs = Input(shape=(photos_per_story, words_per_caption), name="text_input")

    # Use GloVe embeddings => (None, 5, words_per_caption, 300)
    glove_embeddings = TimeDistributed(get_embedding_layer(words_per_caption, words_to_idx))(inputs)

    # Flatten to (None, words_per_caption * 5, 300)
    lstm_input = Reshape((words_per_caption * photos_per_story, 300), input_shape=(5, words_per_caption, 300))(glove_embeddings)

    # LSTM cell returns size (None, words_per_caption * 5, lstm_size)
    encoding = LSTM(lstm_size, return_sequences=True)(lstm_input)

    # Create the text encoder model
    return Model(inputs, encoding, name="text_encoder")
