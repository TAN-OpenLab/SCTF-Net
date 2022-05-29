from utils.running_gpu import running_gpu
running_gpu()
import tensorflow as tf
import math
import sys



from models.SCTF import VideoVioNet_SCTF

from utils.transformer import LoadDataset
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)



def view_bar(message, num, total):
    rate = num / total
    rate_num = int(rate * 40)
    rate_nums = math.ceil(rate * 100)
    r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
    sys.stdout.write(r)
    sys.stdout.flush()


def training(model_name='sctf', epochs=50, batch_size=32, layer_name='add_3'):
    '''

    :param model_name: 模型名称[sctf]
    :param epochs: 训练轮次
    :param batch_size: 每次训练batch尺寸
    :return:
    '''
    global model
    if model_name == 'sctf':
        model = VideoVioNet_SCTF()



    loss_obj = tf.keras.losses.CategoricalCrossentropy()

    optimizer = tf.keras.optimizers.SGD(learning_rate=0.06, decay=1e-2, momentum=0.9)

    train_acc = tf.keras.metrics.CategoricalAccuracy(name='train_acc')
    test_acc = tf.keras.metrics.CategoricalAccuracy(name='test_acc')
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')


    train_dataset, test_dataset = LoadDataset(batch_size).transformer()

    train_acc_ = []
    train_loss_ = []
    test_loss_ = []
    test_acc_ = []
    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            logits = model(images, training=True)
            loss = loss_obj(tf.one_hot(labels, depth=51), logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        train_acc(tf.one_hot(labels, depth=51), logits)
        train_loss(loss)

    @tf.function
    def test_step(images, labels):
        logits = model(images, training=False)
        test_acc(tf.one_hot(labels, depth=51), logits)
        test_loss(loss_obj(tf.one_hot(labels, depth=51), logits))



    for epoch in range(epochs):
        train_acc.reset_states()
        train_loss.reset_states()
        test_acc.reset_states()
        index = 0
        for images, labels in train_dataset:
            train_step(images, labels)
            index += len(labels)
            view_bar("training ", index, 5746)

        print("")
        index = 0
        for images, labels in test_dataset:
            test_step(images, labels)
            index += len(labels)
            view_bar("testing", index, 1020)

        print("")
        tmp = 'Epoch {}, Acc {}, Train loss {},Test Acc {}, Test loss {}'
        print(tmp.format(epoch + 1,
                         train_acc.result() * 100,
                         train_loss.result(),
                         test_acc.result() * 100,
                         test_loss.result()))
        train_acc_.append(round(train_acc.result().numpy()*100.0, 4))
        train_loss_.append(round(train_loss.result().numpy(), 4))
        test_acc_.append(round(test_acc.result().numpy()*100.0, 4))
        test_loss_.append(round(test_loss.result().numpy(), 4))
    print (train_acc_)
    print (train_loss_)
    print (test_acc_)
    print (test_loss_)


if __name__ == '__main__':
    training(model_name='sctf', epochs=200, batch_size=16)
