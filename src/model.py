from layers import *

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

        self.content = None # (batch_size, hidden_dim)
        self.sequences = None # (batch_size, max_steps)
        self.seq_mask = None # (batch_size, max_steps)
        self.hit_at = None # Number
        self.outputs = None # (batch_size, max_steps)

        self.loss = 0
        self.metric = 0
        self.optimizer = None
        self.opt_op = None

    def _loss(self):
        raise NotImplementedError

    def _metric(self):
        raise NotImplementedError

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        self._loss()
        self._metric()

        self.opt_op = self.optimizer.minimize(self.loss)


    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)

class CascadeRNN(Model):
    def __init__(self, number_of_nodes, hidden_dim, max_steps, adjacency_matrix, placeholders, **kwargs):
        super(CascadeRNN, self).__init__(**kwargs)

        self.number_of_nodes = number_of_nodes
        self.hidden_dim = hidden_dim
        self.max_steps = max_steps

        self.content = placeholders['contents']
        self.sequences = placeholders['sequences']
        self.seq_mask = placeholders['seq_mask']
        self.hit_at = placeholders['hit_at']

        self.recurrent_layer = RNN(self.number_of_nodes, self.hidden_dim, adjacency_matrix, name='vanillarnn', logging=kwargs.get('logging'))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        with tf.variable_scope(self.name + '_vars'):
            initer = tf.truncated_normal_initializer(stddev=0.01)
            self.vars['whx'] = tf.get_variable('whx',
                                       dtype=tf.float32,
                                       shape=[hidden_dim, number_of_nodes],
                                       regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularization_scale),
                                       initializer=initer)
            self.vars['bx'] = tf.get_variable('bx', 
                               dtype=tf.float32,
                               shape=[number_of_nodes],
                               regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularization_scale),
                               initializer=tf.zeros_initializer())

        self.build()

    def _build(self):
        x = []
        h_prev = self.content
        x_prev = tf.one_hot(self.sequences[:, 0], self.number_of_nodes)
        for step in range(self.max_steps):
            h_next = self.recurrent_layer(x_prev, h_prev)
            x_next = tf.multiply((1 - x_prev), tf.matmul(h_next, self.vars['whx']) + self.vars['bx']) # keeping who are not active already 
            x.append(x_next)
            h_prev = h_next
            x_prev += tf.one_hot(self.sequences[:, step + 1], self.number_of_nodes)
        
        self.outputs = tf.stack(x, axis=1)

    def _loss(self):
        log_probs = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.outputs, labels=self.sequences[:, 1:])
        loss = tf.reduce_sum(tf.multiply(log_probs, tf.cast(self.seq_mask, tf.float32))) / tf.reduce_sum(tf.cast(self.seq_mask, tf.float32))

        self.loss = loss + tf.losses.get_regularization_loss()

    def _metric(self):
        is_in_top_k = tf.reduce_sum(tf.cast(tf.equal(tf.nn.top_k(self.outputs, k=self.hit_at).indices, tf.expand_dims(self.sequences[:, 1:], axis=2)), tf.float32), axis=2)
        self.metric = tf.reduce_sum(tf.multiply(is_in_top_k, tf.cast(self.seq_mask, tf.float32))) / tf.reduce_sum(tf.cast(self.seq_mask, tf.float32))
