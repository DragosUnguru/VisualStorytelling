import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Embedding


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


def glove_embedding_encoder(sentence_len, words_to_idx, output_units):
    # Receive inputs as string representing a sentence
    inputs = Input(shape=(), dtype=tf.string, name="text_input")

    # Vectorize sentence to word tokens based on given dictionary
    vectorized_input = tf.keras.layers.experimental.preprocessing.TextVectorization(
        max_tokens=None,                        # Exclude <OOV> and the reserved 0 as the output_mode = 'int'
        standardize=None,                       # Input is already standardized
        output_mode='int',
        output_sequence_length=sentence_len,
        vocabulary=list(words_to_idx.keys())    # Exclude <OOV> token as it is automatically added
    )(inputs)

    # Use GloVe embeddings to encode every word before running LSTM
    embeddings = get_embedding_layer(sentence_len, words_to_idx)(vectorized_input)

    # LSTM cell for every word (time step) representing the encoded sentence
    # output = LSTM(output_units)(embeddings)

    # Create the text encoder model
    return Model(inputs, embeddings, name="text_encoder")
