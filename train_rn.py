"""
The relation network tuned for emails
"""
import tensorflow as tf

def model(data, is_training=False):
    """
    data is a batch of tf sparse tensors
    """
    # find padding
    max_len = -1
    for st in data:
        words = tf.string_split(st.values, " ")
        max_len = tf.maximum(words.dense_shape[1], max_len)
        
    # make a fixed length batch of word vectors
    batch_lst = []
    for st in data:
        words = tf.string_split(st.values, " ")
        this_len = words.dense_shape[1]
        batch_lst.append(np.append(np.asarray([glove.get_vector(word) for word in words]), np.zeros([max_len-this_len, glove.EMBEDDING_SIZE]), axis=0))
    batch =

def run():
    data_dir = os.path.expanduser("data")
    
    with tf.Graph().as_default():
        dataset = get_split('train', data_dir)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=32, common_queue_min=1)

        key, content = data_provider.get(['key', 'content'])
        key_batch, content_batch = tf.train.batch(
            [key, content],
            batch_size=FLAGS.batch_size,
            num_threads=FLAGS.num_preprocessing_threads,
            capacity=5 * FLAGS.batch_size,
            dynamic_pad=False)
 
        glove = Glove(os.path.expanduser("~/data/glove/glove.6B.50d.txt"))
        emb = tf.constant(glove.get_nparray())
        glove = None
        # doc_emb is [Batch size x WORD EMBEDDING SIZE]
        doc_emb = embed_document(content_batch, emb)
        neg_doc_emb = embed_document(generate_neg_samples(content_batch), emb)
        net = model(document_emb, is_training)

        # reduce loss
        loss = tf.maximum(0, 1-doc_emb*net + neg_doc_emb*net)        
        
        with tf.Session() as sess:
            with slim.queues.QueueRunners(sess):
                for i in xrange(10):
                    k, c = sess.run([key, content])
                    print "Key: %s\n Content: %s\n" % (k, c)
                    print tf.nn.embedding_lookup(emb, content.values).eval()
