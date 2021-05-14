import tensorflow as tf
from tensorflow.keras.layers import Layer


class TimestepSliceLayer(Layer):
    def __init__(self, **kwargs):
        super(TimestepSliceLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.unstack(inputs, axis=1)

    def get_config(self):
        return super(TimestepSliceLayer, self).get_config()


class EuclidianDistance(Layer):
    def __init__(self, **kwargs):
        super(TimestepSliceLayer, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        return tf.keras.sum(tf.keras.abs(x), axis=-1, keepdims=True)

    def get_config(self):
        return super(TimestepSliceLayer, self).get_config()
