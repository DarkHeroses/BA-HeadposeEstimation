import os
from pprint import pprint

import numpy as np
import pandas as pd
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)
directory = 'C:\\Users\\Hermann\\Documents\\6. Semester - Bachelorarbeit\\Data_Sets\\Biwi\\kinect_head_pose_db\\hpdb\\'


def _parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_png(image_string)
    image_resized = tf.image.resize_images(image_decoded, [64, 64])
    return image_resized, label


def get_datasets():
    # train_data, train_labels, test_data, test_labels, eval_data, eval_label = tf.data.Dataset()
    filenames = []
    labels = []
    folder_counter = sum([len(d) for r, d, folder in os.walk(directory)])
    for i in range(1, 2):  # folder_counter+1):
        print("i" + str(i))
        subdirect = directory + '{:02}'.format(i) + "\\"
        try:
            for filename in os.listdir(subdirect):
                if filename.endswith("_pose.txt"):
                    labels.append(np.reshape(
                        pd.read_csv(os.path.join(subdirect, filename), delimiter=" ", header=None, nrows=3).dropna(
                            axis=1).values, [-1]))
                    continue
                if filename.endswith(".png"):
                    filenames.append(os.path.join(subdirect, filename))
                    continue
                if filename.endswith("_depth.bin"):
                    pass
                else:
                    # print("Lets Continue")
                    continue
        except Exception:
            print("Folder not found")
            continue
    batch_size = 100
    filename_tensor = tf.constant(filenames)
    labels_tensor = tf.constant(np.array(labels))

    dataset = tf.data.Dataset.from_tensor_slices((filename_tensor, labels_tensor))
    dataset = dataset.map(_parse_function)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size)
                            )
    return dataset


def cnn_model_fn(features, labels, mode):
    pprint(features)
    input_layer = tf.reshape(features, [-1, 64, 64, 4])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=30,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    maxp1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=maxp1,
        filters=30,
        kernel_size=[4, 4],
        padding="same",
        activation=tf.nn.relu
    )
    maxp2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=maxp2,
        filters=30,
        kernel_size=[4, 4],
        padding="same",
        activation=tf.nn.relu
    )

    maxp3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(
        inputs=maxp3,
        filters=30,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )
    conv5 = tf.layers.conv2d(
        inputs=conv4,
        filters=120,
        kernel_size=[3, 3],
        padding="same",
        activation=tf.nn.relu
    )
    unit_count = 8*8*120
    flattened = tf.reshape(conv5, [-1, unit_count])
    dense = tf.layers.dense(inputs=flattened, units=unit_count, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    full_layer2 = tf.layers.dense(inputs=dropout, units=84)
    logits = tf.layers.dense(inputs=full_layer2, units=9)

    predictions = {
        'probabilities': tf.nn.sigmoid(logits, name="sigmoid_tensor"),
        'logits': logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=logits)}

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused):
    print('Create the Estimator')
    head_pose_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="C:\\Users\\Hermann\\PycharmProjects\\BachelorArbeit_Headpose Estimation\\tmp\\model")

    print('Set up logging for predictions')
    print('Log the values in the "Sigmoid" tensor with label "probabilities"')
    tensors_to_log = {"probabilities": "sigmoid_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    print('Train the model')
    head_pose_classifier.train(
        input_fn=get_datasets,
        steps=20000,
        hooks=[logging_hook])
    # print("Evaluate the Model")
    # eval_results = head_pose_classifier.evaluate(input_fn=get_datasets)
    # print(eval_results)


if __name__ == "__main__":
    tf.app.run(main)
