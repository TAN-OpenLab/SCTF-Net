import tensorflow as tf

from tensorflow.keras.layers import Dense, Conv3D, AveragePooling3D, MaxPooling3D, Dropout, concatenate, GlobalAveragePooling3D, BatchNormalization

from tensorflow.keras.regularizers import l2
from tensorflow.keras.applications import Xception
from layers.SCT_Block import SCTF
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class CBM(tf.keras.Model):
    def __init__(self, filters, kernel=(1, 1, 1), stride=(1, 1, 1), padding='same'):
        super(CBM, self).__init__()

        self.conv3d = Conv3D(filters, kernel,
                             strides=stride,
                             kernel_initializer='he_normal',
                             kernel_regularizer=l2(0.005),
                             padding=padding)
        self.bn = BatchNormalization()

    def mish(self, x):
        return x * tf.nn.tanh(tf.nn.softplus(x))

    def call(self, input_tensor, training=True):
        x = self.conv3d(input_tensor)
        x = self.bn(x, training=training)
        x = self.mish(x)

        return x


class VideoVioNet_SCTF(tf.keras.Model):

    def __init__(self, layer_name='add_3'):

        super(VideoVioNet_SCTF, self).__init__()
        self.weight_decay = 0.005
        self.layer_name = layer_name
        self.basemodel = Xception(
            weights='imagenet',
            input_shape=(224, 224, 3),
            include_top=False
        )

        # self.basemodel = InceptionResNetV2(
        #     weights='imagenet',
        #     input_shape=(224, 224, 3),
        #     include_top=False
        # )
        # self.basemodel = VGG19(
        #     weights='imagenet',
        #     input_shape=(224, 224, 3),
        #     include_top=False
        # )
        # self.basemodel = ResNet50(
        #     weights='imagenet',
        #     input_shape=(224, 224, 3),
        #     include_top=False
        # )
        # self.basemodel = InceptionV3(
        #     weights='imagenet',
        #     input_shape=(224, 224, 3),
        #     include_top=False
        # )
        # add_3 ~ add_10 [14, 14, 728]
        self.backbone = tf.keras.Model(inputs=self.basemodel.input,
                                       outputs=self.basemodel.get_layer(self.layer_name).output)
        self.backbone.trainable = False

        self.cbm1 = CBM(filters=364, kernel=(1, 1, 2))

        # 14x14x32
        self.block2 = SCTF(32)
        self.avgpooling3d_2 = AveragePooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')

        # 7x7x8
        self.block3 = SCTF(32)
        self.avgpooling3d_3 = AveragePooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')

        # 4x4x4
        self.block4 = SCTF(64)
        self.avgpooling3d_4 = AveragePooling3D((2, 2, 2), strides=(2, 2, 2), padding='same')

        # 2x2x2
        self.cbm5 = CBM(filters=128, kernel=(2, 2, 2), padding='valid')
        self.globalpooling3d_5 = GlobalAveragePooling3D()

        self.dense_2 = Dense(512, activation='relu', kernel_regularizer=l2(self.weight_decay))
        self.dropout_2 = Dropout(0.5)
        # violent video

        # ucf
        self.output_tensor = Dense(51, activation='softmax', kernel_regularizer=l2(self.weight_decay))

    def mish(self, x):
        return x * tf.nn.tanh(tf.nn.softplus(x))

    def call(self, input_tensor, training=True):
        single_frames = []
        for i in range(15):
            frame = self.backbone(input_tensor[:, :, :, i, :], training=False)
            frame = tf.expand_dims(frame, axis=3)
            single_frames.append(frame)
        x = concatenate(single_frames, axis=3)
        x = self.mish(x)

        x = self.cbm1(x, training=training)

        x = self.block2(x, training=training)
        x = self.avgpooling3d_2(x)

        x = self.block3(x, training=training)
        x = self.avgpooling3d_3(x)

        x = self.block4(x, training=training)
        x = self.avgpooling3d_4(x)

        x = self.cbm5(x, training=training)

        x = self.globalpooling3d_5(x)

        x = self.dense_2(x)
        x = self.dropout_2(x)

        return self.output_tensor(x)




