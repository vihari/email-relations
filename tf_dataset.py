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
    'content': 'All of conversation with documents separated by appropriate marker',
    'length': 'The length of content in number of words'
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
        'content': tf.VarLenFeature(tf.int64),
        'length': tf.FixedLenFeature([], tf.int64)
    }

    class Decoder(DataDecoder):
        def __init__(self, keys_to_features):
            self._keys_to_features = keys_to_features

        def decode(self, data, items):
            example = parsing_ops.parse_single_example(data, self._keys_to_features)
            for k in self._keys_to_features:
                v = self._keys_to_features[k]
                if isinstance(v, parsing_ops.FixedLenFeature):
                    example[k] = tf.reshape(example[k], v.shape)

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
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS)

import os
from glove import *
if __name__=='__main__':
    data_dir = os.path.expanduser("data")
    
    with tf.Graph().as_default():
        dataset = get_split('train', data_dir)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=32, common_queue_min=1)

        key, content, length = data_provider.get(['key', 'content', 'length'])
 
        #glove = Glove(os.path.expanduser("~/data/glove/glove.6B.50d.txt"))
        #emb = tf.constant(glove.get_nparray())
        glove = None
        
        with tf.Session() as sess:
            with slim.queues.QueueRunners(sess):
                for i in xrange(10):
                    k, c, l = sess.run([key, content, length])
                    print "Key: %s\n Content: %s Length: %d\n" % (k, c, l)
                    #print tf.nn.embedding_lookup(emb, content.values).eval()
