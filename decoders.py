import slicer_layer
import tensorflow as tf

from tensorflow.keras.layers import Dense, LSTM, Input, Bidirectional, RepeatVector, Concatenate, Activation, Dot, TimeDistributed, Masking, Layer, Reshape
from tensorflow.keras.backend import variable
from tensorflow.keras.activations import softmax
from tensorflow.keras import Model


def softmax_axis_1(x):
    return softmax(x, axis=-1)


def __one_step_attention(a, s_prev, repeator, concatenator, densor1, densor2, activator, dotor):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.

    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, in_seq_len, 2*pre_att_LSTM_size)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, post_att_LSTM_size)

    Returns:
    context -- context vector, input of the next (post-attention) LSTM cell
    """

    # Use repeator to repeat s_prev to be of shape (m, num_words, post_att_LSTM_size)
    # so that we can concatenate it with all hidden states "a" of the pre-attention LSTMs
    s_prev = repeator(s_prev)

    # Use concatenator to concatenate a and s_prev on the last axis
    concat = concatenator([a, s_prev])

    # Use densor1 to propagate concat through a small fully-connected neural network
    # to compute the "intermediate energies" variable e
    e = densor1(concat)

    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies"
    energies = densor2(e)

    # Use "activator" to compute the attention weights "alphas"
    alphas = activator(energies)

    # Use dotor together with "alphas" and "a" to compute the context vector
    # to be given to the next post-attention LSTM-cell
    context = dotor([alphas, a])

    return context


def get_decoder_model_with_attention(
        in_seq_len, out_seq_len, num_features_per_seq_in, pre_att_LSTM_size, post_att_LSTM_size, vocab_size
):
    cell_init = [0] * post_att_LSTM_size
    k_constants = variable(cell_init)
    x = Input(shape=(in_seq_len, num_features_per_seq_in))
    s0 = Input(tensor=k_constants, name='s0')
    c0 = Input(tensor=k_constants, name='c0')
    # s0 = Input(shape=(post_att_LSTM_size,), name='s0')
    # c0 = Input(shape=(post_att_LSTM_size,), name='c0')

    # Don't redefine layers used for attention at every step of attention
    # Define once and pass as argument
    repeator = RepeatVector(in_seq_len)
    concatenator = Concatenate(axis=-1)
    densor1 = Dense(10, activation="tanh")
    densor2 = Dense(1, activation="relu")
    activator = Activation(softmax_axis_1, name='attention_weights')
    dotor = Dot(axes=1)

    s = s0
    c = c0

    # Initialize empty list of outputs
    outputs = []

    # Define pre-attention Bi-LSTM
    a = Bidirectional(LSTM(pre_att_LSTM_size, return_sequences=True, input_shape=(in_seq_len, num_features_per_seq_in)))(x)

    # Post-attention LSTM
    post_att_LSTM = LSTM(post_att_LSTM_size, return_state=True)

    # Softmax output layer
    softmax_layer = Dense(vocab_size, activation=softmax_axis_1)

    # Step 2: Iterate for Ty steps
    for t in range(out_seq_len):
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = __one_step_attention(a, s, repeator, concatenator, densor1, densor2, activator, dotor)

        # Post-attention LSTM cell to the "context" vector
        s, _, c = post_att_LSTM(context, initial_state=[s, c])

        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = softmax_layer(s)

        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)

    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[x, s0, c0], outputs=[outputs, s, c])

    return model


def get_decoder_bilstm(
        photos_per_story, words_per_caption, input_features_len, lstm_size, vocab_size
):

    # Input (None, 5, input_features_len)
    x_in = Input(shape=(photos_per_story, input_features_len))
    final_story_softmax_activations = []

    # Don't train separate LSTM networks
    bi_lstm = Bidirectional(LSTM(lstm_size, return_sequences=True))
    lstm = LSTM(lstm_size, return_sequences=True, return_state=True)

    # Misc for creating story sequence for each photo
    reshaper = Reshape(target_shape=(1, words_per_caption, vocab_size))
    concatenator = Concatenate(axis=1)

    # No initial state for first tensor
    previous_lstm_output = None
    previous_lstm_cell = None

    # Unstack to list of (None, input_features_len) slices
    listed_tensors = slicer_layer.TimestepSliceLayer()(x_in)

    for photo_tensor in listed_tensors:
        # Create num_words_per_caption sequences => tensors of shape (None, words_per_caption, input_features_len)
        seq_photo_tensor = RepeatVector(words_per_caption)(photo_tensor)

        if previous_lstm_output is not None and previous_lstm_cell is not None:
            # First layer Bi-LSTM with initial state from previous run on slice
            initial_state = [previous_lstm_output, previous_lstm_cell, previous_lstm_output, previous_lstm_cell]
        else:
            # Zeros as initial state
            initial_state = None

        # First layer Bi-LSTM
        x = bi_lstm(seq_photo_tensor, initial_state=initial_state)

        # Second layer LSTM
        lstm_out_seq, previous_lstm_output, previous_lstm_cell = lstm(x)

        # Softmax output layer => (None, words_per_caption, vocab_size)
        softmax_activations = TimeDistributed(Dense(vocab_size, activation='softmax'))(lstm_out_seq)
        softmax_activations = reshaper(softmax_activations)
        final_story_softmax_activations.append(softmax_activations)

    # (None, 5, words_per_caption, vocab_size)
    outputs = concatenator(final_story_softmax_activations)

    # Step 3: Create model instance taking three inputs and returning the list of outputs
    model = Model(inputs=x_in, outputs=outputs)

    return model
