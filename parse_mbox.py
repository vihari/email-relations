"""
Parse an mbox and write the dataset as tfrecords
"""
import mailbox
import os
import fnmatch
import re
# make sure punct tokenizer models are installed with nltk.download()
from nltk.tokenize import word_tokenize

import tensorflow as tf
import sys
from glove import *

MBOX_FOLDER = os.path.expanduser("~/data/epadd/Bush small 2/")
_NUM_SHARDS = 5
# used to separate one doc from other
DOC_SEP_STR = " %%%%%%%%%% "
DATASET_NAME = "bush-small"

mbox_files = []
for path, dirs, filenames in os.walk(MBOX_FOLDER):
    mbox_files += [os.path.join(path, f) for f in fnmatch.filter(filenames, '*.mbox')]

def get_addr(string):
    """
    "Jeb Bush" <jeb@jeb.org> -> jeb@jeb.org
    
    returns only one of several addresses in the string
    "McFadden, Liza" <liza.mcfadden@myflorida.com>,\r\n   "Steele, Leslie" <leslie.steele@myflorida.com>,\r\n   "Pfeifer, Pam" <pam.pfeifer@laspbs.state.fl.us>,\r\n   "Cole, Cristal" <cristal.cole@myflorida.com> -> liza.mcfadden@myflorida.com
    """
    x = get_addr.patt.search(string)
    if not x:
       sys.stderr.write("Unexpected address string %s\n" % string)
       return None
    return string[x.start():x.end()]
    
get_addr.patt = re.compile("<.*?>")

# Returns the content ignoring the original message
def ignore_original(content):
    match = ignore_original.patt.search(content)
    if match:
        return content[:match.start()]
    return content

ignore_original.patt = re.compile("\n.*?original message", re.IGNORECASE)

class Document:
    def __init__(self, words):
        self._words = words
    
    def __str__(self):
        return " ".join(self._words)
        
    def retain_vocab(self, vocab):
        self._words = [w for w in self._words if w in vocab]

    @property    
    def num_words(self):
        return len(self._words)
        
vocab = {}
conv = {}
num_ignored = 0
total = 0
for mf in mbox_files:
    sys.stderr.write("Reading MBOX file %s\n" % mf)
    for message in mailbox.mbox(mf):
        total += 1
        if message.is_multipart():
            num_ignored += 1
            continue

        from_addr = get_addr(message["from"])
        to_addr = get_addr(message["to"])
        content = message.get_payload(decode=True)
        if not content or not from_addr or not to_addr:
            num_ignored += 1
            continue

        content = ignore_original(content)
        try:
            words = word_tokenize(content)
        except UnicodeDecodeError:
            num_ignored += 1
            continue
        for w in words:
            vocab[w] = vocab.get(w, 0) + 1
        key = from_addr+'->'+to_addr
        conv[key] = conv.get(key, []) + [Document(words)]

sys.stderr.write("Ignored %d of total %d messages\n" % (num_ignored, total))
sys.stderr.write("Total number of words %d\n" % len(vocab))
vocab = vocab.items()
vocab.sort(lambda x, y: cmp(x[1], y[1]))
# remove the most frequent third of the vocabulary
# vocab = vocab[:2*int(len(vocab)/3.)]
# throw away single frequency words
vocab = [x for x in vocab if x[1]>1]
sys.stderr.write("Number of words is reduced to %d\n" % len(vocab))
# can throw the frequency away now
vocab, _ = zip(*vocab)
for k in conv.keys():
    for doc in conv[k]:
        doc.retain_vocab(set(vocab))

# Write TFRecords
import math

data = conv.items()
save_dir = "data"
num_per_shard = int(math.ceil(len(data) / float(_NUM_SHARDS)))
num_keys = 0
glove = Glove(os.path.expanduser("~/data/glove/glove.6B.50d.txt"))
with tf.Graph().as_default():
    with tf.Session('') as sess:

        for shard_id in range(_NUM_SHARDS):
            output_filename = os.path.join(save_dir, "%s-train-%d-of-%d.tfrecord" %
                                           (DATASET_NAME, shard_id, _NUM_SHARDS))
            with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                start_ndx = shard_id * num_per_shard
                end_ndx = min((shard_id + 1) * num_per_shard, len(data))
                for i in range(start_ndx, end_ndx):
                    sys.stdout.write('\r>> Writing conversation %d/%d shard %d' % (
                        i + 1, len(data), shard_id))
                    sys.stdout.flush()

                    # drop the key if the total words is less than threshold
                    total_words = sum([doc.num_words for doc in data[i][1]])
                    docs = [doc for doc in data[i][1] if doc.num_words>3]
                    if total_words < 10 or len(docs) == 0:
                        continue

                    num_keys += 1
                    content = DOC_SEP_STR.join([str(doc) for doc in docs])
                    coded_content = [glove.vocab[word] for word in content.lower().split() if word in glove.vocab]
                                        
                    example = tf.train.Example(features=tf.train.Features(feature={
                        'key': tf.train.Feature(bytes_list=tf.train.BytesList(
                            value=[data[i][0]])),
                        'content': tf.train.Feature(int64_list=tf.train.Int64List(value=coded_content))
                    }))

                    tfrecord_writer.write(example.SerializeToString())
                    
sys.stdout.write('\n')
sys.stdout.flush()
sys.stdout.write("Wrote %d total keys to the dataset\n" % num_keys)
