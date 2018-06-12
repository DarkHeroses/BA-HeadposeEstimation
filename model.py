import numpy as np
import tensorflow as tf
import os
import pandas as pd
import sys

tf.logging.set_verbosity(tf.logging.INFO)
directory = 'C:\\Users\\Hermann\\Documents\\6. Semester - Bachelorarbeit\\Data_Sets\\Biwi\\kinect_head_pose_db\\hpdb\\'


def read_label_file(temp_dir):
    # print(temp_dir)
    file = open(temp_dir, "r")
    mat = np.zeros([4, 3], dtype=float)
    # print(mat)
    i = 0
    j = 0
    for line in file:
        # print("i: " + str(i))
        if line == "\n":
            continue
        array_ofstring = line.split()
        for el in array_ofstring:
            # print("j: " + str(j))
            mat[i][j] = float(el)
            j += 1
        i += 1
        j = 0
    # print(mat)

    file.close()

    return mat


def get_datesets():
    counter = 0
    images = []
    labels = []
    depth_images = []
    folder_counter = sum([len(d) for r, d, folder in os.walk(directory)])
    datasets = {"rgb": None, "label": None, "depth": None}
    for i in range(folder_counter):
        subdirect = directory + '{:02}'.format(i + 1) + "\\"
        # print("Subdirectory Var created:" + subdirect)
        try:
            for filename in os.listdir(subdirect):
                if filename.endswith("_pose.txt"):
                    labels.append(tf.reshape(read_label_file(os.path.join(subdirect, filename)), [-1]))
                if filename.endswith(".png"):
                    # print("looking for files")
                    # #print(os.path.join(directory, filename))
                    images.append(tf.image.decode_png(tf.read_file(os.path.join(subdirect, filename)), channels=1))
                    counter += 1
                    # Dateipfad + '{:02}'.format(i) + "\\frame_" + '{:05}'.format(j) + "_rgb.png"

                    # print("added to dataset")
                if filename.endswith("_depth.bin"):
                    # print("looking for files")
                    # #print(os.path.join(directory, filename))
                    depth_images.append(
                        tf.image.decode_png(tf.read_file(os.path.join(subdirect, filename)), channels=1))
                    # Dateipfad + '{:02}'.format(i) + "\\frame_" + '{:05}'.format(j) + "_rgb.png"
                    # print("added to dataset")
                else:
                    # print("Lets Continue")
                    continue
        except Exception:
            # print("Folder not found")
            continue
    # print("Folder_count:" + str(folder_counter))
    # print("Files added:" + str(counter))
    print("RGB")
    datasets["rgb"] = tf.data.Dataset.from_tensors(images)
    print("label")
    datasets["label"] = tf.data.Dataset.from_tensor_slices(labels)
    print("depth")
    datasets["depth"] = tf.data.Dataset.from_tensor_slices(depth_images)
    return datasets


def cnn_model_fn(features, labels, mode):
    input_layer = features["x"]  # tf.reshape(features["x"], [-1, 640, 480, 3])

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
    dense = tf.layers.dense(inputs=conv5, units=120, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
    full_layer2 = tf.layers.dense(inputs=dropout, units=84)
    logits = tf.layers.dense(inputs=full_layer2, units=12)
    predicted_classes = tf.argmax(logits, 1)

    predictions = {
        'class_ids': predicted_classes[:, tf.newaxis],
        'probabilities': tf.nn.softmax(logits),
        'logits': logits,
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predicted_classes)}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused):
    # sess = tf.InteractiveSession()
    biwi = get_datesets()
    print('Create the Estimator')
    head_pose_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir="C:\\Users\\Hermann\\PycharmProjects\\BachelorArbeit_Headpose Estimation\\tmp\\model")

    print('Set up logging for predictions')
    print('Log the values in the "Softmax" tensor with label "probabilities"')
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    print('Train the model')
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": biwi["rgb"]},
        y=biwi["label"],
        batch_size=100,
        num_epochs=None,
        shuffle=True)
    print("head_pose_classifier.train")
    head_pose_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook])

    print('Evaluate the model and print results')
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": biwi["rgb"][0]},
        y=biwi["label"],
        num_epochs=1,
        shuffle=False)

    eval_results = head_pose_classifier.evaluate(input_fn=eval_input_fn)
    file = open("eval_results.log", "w")
    print(eval_results)
    file.write(eval_results)
    file.close()


if __name__ == "__main__":
    tf.app.run()
