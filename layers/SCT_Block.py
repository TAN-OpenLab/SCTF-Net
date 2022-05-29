import tensorflow as tf

from tensorflow.keras.layers import Conv3D, add, BatchNormalization

from tensorflow.keras.regularizers import l2



class SCTF(tf.keras.layers.Layer):
    def __init__(self, filters=32, kernel_size=(3, 3, 3), strides=(1, 1, 1), weight_decay=0.005):
        super(SCTF, self).__init__()

        self.filters = filters

        self.kernel_size = kernel_size
        self.strides = strides
        self.weight_decay = weight_decay

    def build(self, input_shape):
        """

        :param input_shape: [None, W, H, T, C]
        :return:
        """
        super(SCTF, self).build(input_shape)

        self.T = input_shape[3]

        self.conv3d_space = Conv3D(
            filters=self.T,
            kernel_size=(self.kernel_size[0], self.kernel_size[1], 1),
            strides=self.strides,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(self.weight_decay),
        )
        self.conv3d_1 = Conv3D(
            filters=self.filters,
            kernel_size=(1, 1, 1),
            strides=self.strides,
            padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(self.weight_decay),
        )
        self.conv3d_temporal = Conv3D(
            filters=self.filters,
            kernel_size=(1, 1, self.T),
            strides=(1, 1, 1),
            padding='valid',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(self.weight_decay),
        )
        self.bn = BatchNormalization()

    def mish(self, tensor):
        return tensor*tf.nn.tanh(tf.nn.softplus(tensor))

    def call(self, input_tensor, training=True):

        input_space = tf.transpose(input_tensor, (0, 1, 2, 4, 3))
        out_space = self.mish(self.conv3d_space(input_space))
        out_space = tf.transpose(out_space, (0, 1, 2, 4, 3))
        # print(out_space.shape)
        out_space = self.conv3d_1(out_space)
        out_temporal = self.mish(self.conv3d_temporal(input_tensor))
        out_full_temporal = tf.tile(out_temporal, multiples=[1, 1, 1, self.T, 1])
        out = add([out_space, out_full_temporal])
        out = self.bn(out, training=training)
        out = self.mish(out)
        return out



