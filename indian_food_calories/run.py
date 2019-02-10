import logging
import warnings
from math import floor
import tensorflow as tf
import calendar
import time

from models.data_loader import ImageLoader
from tools.image_loader import load_image

# Settings
EPOCH_COUNT = 200
BATCH_SIZE = 256
BATCH_COUNT = 100
IMAGE_SIZE = 124
TRAINING = True
GLOBAL_STEP = 0

# Setup
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR, format='%(name)s(%(levelname)s) - %(message)s')

loader = ImageLoader(transform_fn=load_image, test_size=1000, im_size=IMAGE_SIZE)
if BATCH_COUNT * BATCH_SIZE > len(loader.train_data):
    BATCH_COUNT = floor(len(loader.train_data) / BATCH_SIZE)

# Tensorflow Definitions
with tf.name_scope("io"):
    X = tf.placeholder(dtype=tf.float32, shape=[None, IMAGE_SIZE, IMAGE_SIZE, 3], name="inputs")
    y = tf.placeholder(dtype=tf.int8, shape=[None, 50], name="labels")

# Convolution Block
def conv_block(index, inputs):
        conv_output = tf.layers.conv2d(inputs=inputs, filters=(2 ** (index + 1)) * 8, kernel_size=(5, 5),
                                       activation=tf.nn.relu, name="conv_{0}".format(index), trainable=TRAINING)
        pool_output = tf.layers.max_pooling2d(inputs=conv_output, pool_size=(2, 2), strides=2,
                                              name="pooling_{0}".format(index))
        dropped_output = tf.layers.dropout(pool_output, training=TRAINING, rate=0.4)
        normed_output = tf.layers.batch_normalization(dropped_output, axis=-1, training=TRAINING,
                                                      name="batch_norm_{0}".format(index))
        return normed_output


# Add Feature Extractor Layers
current_output = X
with tf.name_scope("conv_layers"):
    for i in range(1, 5):
        current_output = conv_block(i, current_output)
        tf.summary.histogram("block{0}_outputs".format(i), current_output)

# Add Dense Layers
with tf.name_scope("dense_layers"):
    flattened_features = tf.reshape(current_output, [-1, 4 * 4 * 256], name="flattened")
    dense_output1 = tf.layers.dense(inputs=flattened_features, units=1024, activation=tf.nn.relu, name="dense_1", trainable=TRAINING)
    tf.summary.histogram("dense1_outputs", dense_output1)
    dense_output2 = tf.layers.dense(inputs=dense_output1, units=256, activation=tf.nn.relu, name="dense_2", trainable=TRAINING)
    tf.summary.histogram("dense2_outputs", dense_output2)
    dense_output3 = tf.layers.dense(inputs=dense_output2, units=50, activation=tf.nn.relu, name="predictions", trainable=TRAINING)
    tf.summary.histogram("dense3_outputs", dense_output3)


# Loss Calculation
with tf.name_scope("loss"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=dense_output3, labels=y))
    train_op = tf.train.AdamOptimizer().minimize(loss=loss)
    tf.summary.scalar("Loss", loss)

# Accuracy Calculation
with tf.name_scope("accuracy"):
    predictions = tf.argmax(dense_output3, 1)
    correct = tf.equal(tf.argmax(y, 1), predictions)
    acc_op = tf.reduce_mean(tf.cast(correct, "float"))
    tf.summary.scalar("accuracy", acc_op)

# Load Test Data
print("Preparing Test Data...\n")
test_image_infos = loader.test_data[:]
test_image_data = [info["image"] for info in test_image_infos]
test_image_label = [info["label"] for info in test_image_infos]


with tf.Session() as sess:

    train_writer = tf.summary.FileWriter("./log/tensorboard/train", sess.graph)
    test_writer = tf.summary.FileWriter("./log/tensorboard/test")
    summaries = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())

    for j in range(EPOCH_COUNT):
        for i in range(BATCH_COUNT):
            TRAINING = True
            print("\033[1;32;49m Epoch {0} - Batch {1}:".format(j, i))
            train_image_infos = loader.train_data[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            train_image_data = [info["image"] for info in train_image_infos]
            train_image_label = [info["label"] for info in train_image_infos]
            train_summs, train_out, train_acc_out, train_loss_out = sess.run([summaries, train_op, acc_op, loss],
                                                           feed_dict={X: train_image_data, y: train_image_label})
            print("\033[0;37;49m Training Accuracy: {:.2f}%, Training Loss: {:.2f}".format(train_acc_out * 100, train_loss_out))

            train_writer.add_summary(train_summs, global_step=GLOBAL_STEP)

            TRAINING = False
            test_summs, test_acc_out, test_loss_out = sess.run([summaries, acc_op, loss],
                                                           feed_dict={X: test_image_data, y: test_image_label})
            print("\033[2;35;49m Test Accuracy: {:.2f}%\033[0;35;49m, Test Loss: {:.2f}\n".format(test_acc_out * 100, test_loss_out))
            test_writer.add_summary(test_summs, global_step=GLOBAL_STEP)
            GLOBAL_STEP+=1
        loader.reshuffle()

        if j % 5 == 0:
            # Save model every 5 epochs
            ts = calendar.timegm(time.gmtime())
            tf.saved_model.simple_save(sess, "trained_models/acc_{:.2f}_epoch_{:d}_{:d}".format(test_acc_out * 100, j, ts), inputs= {
                "X": X, "y": y
            }, outputs= {
                "z": dense_output3
            })
