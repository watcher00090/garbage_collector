import numpy as np
import tensorflow as tf
import pandas as pd

def train_input_fn(features, labels, batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(({"pixel_matrices" : np.array(features)},np.array(labels)))
    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)

def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features={"pixel_matrices" : np.array(features)}
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, np.array(labels))

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

from scipy import misc
from random import shuffle

training_set_fraction = .7
batch_size = 20
num_epochs = 2

dataandlabels = []

f = open("labels.txt", "r")
contents = f.read()
lines = contents.split("\n")
for i in range(0,len(lines)-1): #-1 so that we don't pick up the newline at the end of the file
    parts = lines[i].split(",")
    dataandlabels.append((misc.imread('trash/' + parts[0]), int(parts[1])))
f.close()

matrix_feature_column = tf.feature_column.numeric_column(key="pixel_matrices", shape=[256,256,3])

classifier = tf.estimator.DNNClassifier(
    feature_columns=[matrix_feature_column],
    hidden_units=[100,100,100,100,100,100,100],
    # The model must choose between 2 classes.
    # model_dir = "model_dir",
    n_classes=2)

f = open('test_results.txt', 'w')

for p in range(0, num_epochs):
    data = []
    labels = []
    train_x = []
    train_y = []
    test_x = []
    test_y = []

    shuffle(dataandlabels)
                              
    for i in range(0, len(dataandlabels)):
        data.append(dataandlabels[i][0])
        labels.append(dataandlabels[i][1])
                              
    train_size = int(np.floor(len(data) * training_set_fraction))

    for i in range(0, train_size):
        train_x.append(data[i])
        train_y.append(labels[i])

    for i in range(0, len(data) - int(train_size)):
        test_x.append(data[train_size + i])
        test_y.append(labels[train_size + i])

    classifier.train(
        input_fn=lambda:train_input_fn(train_x, train_y, batch_size),
        steps=10)

    eval_result = classifier.evaluate(
            input_fn=lambda:eval_input_fn(test_x, test_y,
                                                    batch_size))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    f.write(str(eval_result['accuracy']) + "\n");

    #var = [v for v in tf.trainable_variables() if v.name == "tower_2/filter:0"][0]
    print(tf.trainable_variables())



f.close()
