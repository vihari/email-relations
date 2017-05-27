"""
The relation network tuned for emails
"""
import tensorflow as tf
import os
import glove
import tf_dataset

slim = tf.contrib.slim
tf.app.flags.DEFINE_float('learning_rate', .1,
                           'The learning rate')
tf.app.flags.DEFINE_string('logdir', os.path.expanduser("~/sandbox/rmn_epadd/"),
                           'The directory to put checkpoints and summary')
tf.app.flags.DEFINE_integer('num_relation', 5,
                            'The number of categories for relation')
tf.app.flags.DEFINE_integer('batch_size', 32,
                            'The batch size')
tf.app.flags.DEFINE_string('data_dir', os.path.expanduser("data"),
                           'The folder where the training data resides')
tf.app.flags.DEFINE_string('embeddings', os.path.expanduser("~/data/glove/glove.6B.50d.txt"),
                           'The file containing embedding vectors')
tf.app.flags.DEFINE_integer('log_every_n_steps', 10,
                           'The frequency with which the loss is printed')
tf.app.flags.DEFINE_integer('number_of_steps', 1000,
                            'Number of training steps')
FLAGS = tf.app.flags.FLAGS

def document_embedding(doc_batch, lengths, word_emb):
    """Batch of tf sparse tensors"""
    emb_size = glove.EMBEDDING_SIZE
    # lstm_cell = tf.contrib.rnn.LSTMCell(emb_size)

    doc_vector = tf.nn.embedding_lookup(word_emb, doc_batch)
    # net, _ = tf.nn.dynamic_rnn(lstm_cell, doc_vector, sequence_length=lengths, dtype=tf.float64)
    # net is of shape [Batch size x Sequence Length x EMBEDDING SIZE] -- collect the value at the last time slice
    # net = net[:, -1, :]
    # doc_vector[bi, :lengths[bi]]
    net = tf.map_fn(lambda bi: tf.reduce_mean(doc_vector[bi, :lengths[bi]], axis=0), tf.range(tf.shape(doc_batch)[0]), dtype=tf.float64)
    return net

def model(data, keys, gl, is_training=False):
    hidden_width = 20

    # data = tf.Print(data, [tf.nn.moments(data, axes=[1])[1]], message="Variance of batch")
    W_h = tf.get_variable("W_h", initializer=tf.random_normal([glove.EMBEDDING_SIZE, hidden_width], stddev=1E-2, dtype=tf.float64))
    bias_h = tf.get_variable("bias_h", initializer=tf.zeros([hidden_width], dtype=tf.float64))
    net = tf.nn.relu(tf.matmul(data, W_h)+bias_h)
    # net = tf.cast(tf.contrib.layers.batch_norm(tf.cast(net, tf.float32), is_training=True), tf.float64)
    
    W_d = tf.get_variable("W_d", initializer=tf.random_normal([hidden_width, FLAGS.num_relation], stddev=1E-2, dtype=tf.float64))
    bias_d = tf.get_variable("bias_d", initializer=tf.zeros([FLAGS.num_relation], dtype=tf.float64))
    net = tf.matmul(net, W_d)+bias_d
    # net = tf.cast(tf.contrib.layers.batch_norm(tf.cast(net, tf.float32), is_training=True), tf.float64)

    # net = tf.Print(net, [net], message="Net b4 SM")
    net = tf.nn.softmax(net)
    d=data

    import numpy as np
    top = tf.nn.top_k(net)
    # ids = [gl.vocab.get(wrd) for wrd in ['father', 'politics', 'arab', 'federal', 'friend']]
    ids = [gl.vocab.get(wrd) for wrd in ['friend', 'travel', 'research', 'computer', 'bug']]
    R_init = np.asarray([gl.W[_id] for _id in ids], dtype=np.float64)
    # R_init = tf.random_normal([FLAGS.num_relation, glove.EMBEDDING_SIZE], stddev=1, dtype=tf.float64)
    # net = tf.Print(net, [top.values, top.indices, tf.gather(keys, tf.where(tf.equal(tf.squeeze(top.indices), 0)))], message="Strong with first")
    R = tf.get_variable("R", initializer=R_init)
    net = tf.matmul(net, R)
    return net, tf.norm(tf.matmul(R, R, transpose_b=True)-tf.eye(FLAGS.num_relation, dtype=tf.float64)), d
    
def main(_):
    data_dir = FLAGS.data_dir
    tf.logging.set_verbosity(tf.logging.INFO)
    
    with tf.Graph().as_default():
        dataset = tf_dataset.get_split('train', data_dir)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset, common_queue_capacity=32, common_queue_min=1, shuffle=True)

        key, content, length = data_provider.get(['key', 'content', 'length'])
        num_negs = 5
        
        global_step = slim.create_global_step()
        batch_size = FLAGS.batch_size + 5*FLAGS.batch_size
        key_batch, content_batch, length_batch = tf.train.batch(
            [key, content, length],
            batch_size=batch_size,
            capacity=5 * batch_size,
            dynamic_pad=True)
        length_batch = tf.cast(length_batch, tf.int32)
        content_batch = content_batch.values
        # content_batch = tf.Print(content_batch, [content_batch], "tf print")
        max_len = tf.reduce_max(length_batch)
        index = tf.map_fn(lambda i: tf.reduce_sum(length_batch[:i]), tf.range(batch_size), tf.int32)
        # make a Tensor batch
        content_batch = tf.map_fn(lambda i: tf.concat(
            [content_batch[index[i]:index[i]+length_batch[i]],
             tf.zeros([max_len-length_batch[i]], tf.int64)], 0),
                                  tf.range(batch_size), dtype=tf.int64)
        print key_batch.get_shape(), content_batch, length_batch
 
        gl = glove.Glove(FLAGS.embeddings)
        emb = tf.constant(gl.get_nparray())
        # doc_emb is [Batch size x WORD EMBEDDING SIZE]
        with tf.variable_scope("document_embedding"):
            doc_emb = document_embedding(content_batch[ : FLAGS.batch_size], length_batch[ : FLAGS.batch_size], emb)
        with tf.variable_scope("document_embedding", reuse=True):
            neg_doc_emb = document_embedding(content_batch[FLAGS.batch_size: ], length_batch[FLAGS.batch_size: ], emb)
        # net is [Batch size x WORD EMBEDDING SIZE]
        is_training = True
        net, R_penalty, _ = model(doc_emb, key_batch.values, gl, is_training)

        # reduce loss
        neg_doc_emb = tf.reshape(neg_doc_emb, [num_negs, FLAGS.batch_size, -1])
        # neg_doc_emb = tf.map_fn(lambda batch_idx: tf.gather(neg_doc_emb, tf.random_uniform([num_negs], 0, tf.shape(neg_doc_emb)[0], tf.int32)), tf.range(FLAGS.batch_size), dtype=tf.float64)
        loss = tf.reduce_sum(tf.map_fn(lambda ni: tf.maximum(tf.cast(0, tf.float64), 1 + tf.reduce_sum(-doc_emb*net + neg_doc_emb[ni]*net, axis=1)), tf.range(num_negs), tf.float64))
        # loss = tf.reduce_sum(tf.maximum(tf.cast(0, tf.float64), 1 + tf.reduce_sum(- doc_emb*net + tf.reduce_sum(neg_doc_emb, axis=0)*net, axis=1))) 
        loss += .01*R_penalty
        
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=FLAGS.learning_rate)
        train_op = slim.learning.create_train_op(loss, optimizer, clip_gradient_norm=4, global_step=global_step)

        def custom_train_step_fn(sess, train_op, global_step, train_step_kwargs):
            if custom_train_step_fn.step>0 and custom_train_step_fn.step % 100 == 0:
                tf_R = [v for v in tf.global_variables() if v.name=="R:0"][0]
                np_R = sess.run(tf_R)
                descs = gl.get_closest(np_R)
                print "The relations learned so far"
                for i, d in enumerate(descs):
                    print "%2d %s" % (i, d)
                                
            custom_train_step_fn.step += 1
                
            return slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)
        custom_train_step_fn.step=0
        
        slim.learning.train(train_op, FLAGS.logdir,
                            log_every_n_steps=FLAGS.log_every_n_steps,
                            save_interval_secs=100,
                            #number_of_steps=FLAGS.number_of_steps,
                            train_step_fn=custom_train_step_fn
        )

if __name__=='__main__':
    tf.app.run()
