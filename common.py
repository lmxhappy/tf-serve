# -*- coding: utf-8 -*-

import os
import tensorflow as tf

def get_export_dir():
    models = [int(i) for i in os.listdir('export')]
    if not models:
        raise ValueError('no models found')

    export_dir = 'export/{}'.format(max(models))
    print('Latest export dir {}'.format(export_dir))
    return export_dir


def create_examples(inputs):
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
    return examples


def assemble_result(inputs, predictions):
    result = inputs.copy()
    for index, row in result.iterrows():
        scores = predictions['scores'][index]
        score = scores.max()
        class_index = scores.argmax()
        class_id = predictions['classes'][index][class_index].decode()
        result.loc[index, 'ClassID'] = class_id
        result.loc[index, 'Probability'] = score
    return result
