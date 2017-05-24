"""
Dataset provider; reads tfrecords and provides dataset
"""
import os
import tensorflow as tf

from tensorflow.python.ops import parsing_ops
from tensorflow.contrib.slim.python.slim.data.data_decoder import DataDecoder

slim = tf.contrib.slim

SPLITS_TO_SIZES = {'train': 1000}
_FILE_PATTERN = "bush-small-%s-*.tfrecord"

_ITEMS_TO_DESCRIPTIONS = {
    'key': 'The agents involved in the conversation',
    'content': 'All of conversation with documents separated by appropriate marker'
}

def get_split(split_name, dataset_dir, file_pattern=None, reader=None):
    if split_name not in SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    # Allowing None in the signature so that dataset_factory can use the default.
    if reader is None:
        reader = tf.TFRecordReader

    keys_to_features = {
        'key': tf.VarLenFeature(tf.string),
        'content': tf.VarLenFeature(tf.string)
    }

    class Decoder(DataDecoder):
        def __init__(self, keys_to_features):
            self._keys_to_features = keys_to_features

        def decode(self, data, items):
            example = parsing_ops.parse_single_example(data, self._keys_to_features)

            outputs = []
            for item in items:
                outputs.append(example[item])

            return outputs

        def list_items(self):
            return _ITEMS_TO_DESCRIPTIONS.keys()

    return slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=reader,
        decoder=Decoder(keys_to_features),
        num_samples=SPLITS_TO_SIZES[split_name],
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=0,
        labels_to_names={})

if __name__=='__main__':
    data_dir = os.path.expanduser("data")
    
    with tf.Graph().as_default():
        dataset = get_split('train', data_dir)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=32, common_queue_min=1)

        key, content = data_provider.get(
            ['key', 'content'])

        with tf.Session() as sess:
            with slim.queues.QueueRunners(sess):
                for i in xrange(10):
                    k, c = sess.run([key, content])
                    print "Key: %s\n Content: %s\n" % (k, c)
