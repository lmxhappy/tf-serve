# -*- coding: utf-8 -*-

import pandas as pd
import tensorflow as tf
from tensorflow.python.tools import saved_model_utils
from tensorflow.contrib.saved_model.python.saved_model import signature_def_utils
import common


with tf.Session() as sess:
    export_dir = common.get_export_dir()
    tf.saved_model.loader.load(sess, ['serve'], export_dir)

    meta_graph_def = saved_model_utils.get_meta_graph_def(export_dir, 'serve')
    predict_signature_def = signature_def_utils.get_signature_def_by_key(meta_graph_def, 'serving_default')

    inputs = pd.DataFrame({
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    })
    examples = common.create_examples(inputs)

    fetches = [predict_signature_def.outputs[key].name for key in ['classes', 'scores']]
    feed_dict = {predict_signature_def.inputs['inputs'].name: examples}

    outputs = sess.run(fetches, feed_dict=feed_dict)
    predictions = {
        'classes': outputs[0],
        'scores': outputs[1],
    }

    print(common.assemble_result(inputs, predictions))
