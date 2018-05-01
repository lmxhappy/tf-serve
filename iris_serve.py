# -*- coding: utf-8 -*-

import os
import pandas as pd
import tensorflow as tf
import common

# load model
export_dir = common.get_export_dir()
predict_fn = tf.contrib.predictor.from_saved_model(export_dir)

# prepare test features
inputs = pd.DataFrame({
    'SepalLength': [5.1, 5.9, 6.9],
    'SepalWidth': [3.3, 3.0, 3.1],
    'PetalLength': [1.7, 4.2, 5.4],
    'PetalWidth': [0.5, 1.5, 2.1],
})
examples = common.create_examples(inputs)

# predict
predictions = predict_fn({'inputs': examples})

print(common.assemble_result(inputs, predictions))
