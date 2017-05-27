import tensorflow as tf
from train_rn import *

def main(_):
    data_dir = "data" # FLAGS.data_dir
    checkpoint_dir = os.path.expanduser("~/sandbox/rmn_epadd")
    embeddings = os.path.expanduser("~/data/glove/glove.6B.50d.txt")
    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        dataset = tf_dataset.get_split('train', data_dir)
        data_provider = slim.dataset_data_provider.DatasetDataProvider(
            dataset,num_epochs=1, shuffle=False)

        key, content, length = data_provider.get(['key', 'content', 'length'])
        
        global_step = slim.create_global_step()
        batch_size = 32
        # key_batch, content_batch, length_batch = key, content.
        
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

        gl = glove.Glove(embeddings)
        emb = tf.constant(gl.get_nparray())
        doc_emb = document_embedding(content_batch, length_batch, emb)
        net, _, d = model(doc_emb, key_batch.values, gl, False)
        
        f_name = tf.train.latest_checkpoint(checkpoint_dir)
        tf.logging.info('Evaluating %s' % f_name)
        
        with tf.Session().as_default() as sess:
            sess.run(tf.local_variables_initializer())
            coord = tf.train.Coordinator()
            tf.train.start_queue_runners(sess, coord=coord)
            saver = tf.train.Saver()
            saver.restore(sess, save_path=f_name)
            g=0
            with tf.gfile.GFile(os.path.expanduser("~/sandbox/vectors.txt"), mode="w") as f: 
                while True:
                    try:
                        keys, vectors = sess.run([key_batch, d])
                        for i, v in enumerate(keys.values):
                            f.write("%s %s\n" % (keys.values[i].replace(" ", "_"), " ".join(map(str, vectors[i]))))
                        g+=1
                    except tf.errors.OutOfRangeError, e:
                        print "Done"
                        break

tf.app.run()
