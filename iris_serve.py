# -*- coding: utf-8 -*-

import os
import pandas as pd
import tensorflow as tf

# load model
models = [int(i) for i in os.listdir('export')]
if not models:
    raise ValueError('no models found')

export_dir = 'export/{}'.format(max(models))
print('Load model from {}'.format(export_dir))
predict_fn = tf.contrib.predictor.from_saved_model(export_dir)

# prepare test features
inputs = pd.DataFrame({
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
})

examples = []
for index, row in inputs.iterrows():
    feature = {}
    for col, value in row.iteritems():
        feature[col] = tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
    example = tf.train.Example(
        features=tf.train.Features(
            feature=feature
        )
    )
    examples.append(example.SerializeToString())

# predict
predictions = predict_fn({'inputs': examples})

outputs = inputs.copy()
for index, row in outputs.iterrows():
    scores = predictions['scores'][index]
    score = scores.max()
    class_index = scores.argmax()
    class_id = predictions['classes'][index][class_index].decode()
    outputs.loc[index, 'ClassID'] = class_id
    outputs.loc[index, 'Probability'] = score

print(outputs)
