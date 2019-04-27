import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

class RecurrentLayer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging

    def _call(self, x, h_prev):
        raise NotImplementedError

    def __call__(self, x, h_prev):
        with tf.name_scope(self.name):
            if self.logging:
                tf.summary.histogram(self.name + '/x', x)
                tf.summary.histogram(self.name + '/h', h_prev)
            outputs = self._call(x, h_prev)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class RNN(RecurrentLayer):

    def __init__(self, number_of_nodes, hidden_dim, adjacency, **kwargs):
        super(RNN, self).__init__(**kwargs)

        self.number_of_nodes = number_of_nodes
        self.hidden_dim = hidden_dim
        self.adj = adjacency

        with tf.variable_scope(self.name + '_vars'):
            initer = tf.truncated_normal_initializer(stddev=0.01)
            self.vars['wx'] = tf.get_variable('wx',
                                       dtype=tf.float32,
                                       shape=[number_of_nodes, hidden_dim],
                                       regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularization_scale),
                                       initializer=initer)
            self.vars['wh'] = tf.get_variable('wh',
                                       dtype=tf.float32,
                                       shape=[hidden_dim, hidden_dim],
                                       regularizer=tf.contrib.layers.l2_regularizer(FLAGS.regularization_scale),
                                       initializer=initer)


    def _call(self, x, h_prev):
        """
        x: active users (batch_size, number_of_users)
        h_prev: previous state of the network (batch_size, hidden_dim)
        """

        content_vector = tf.matmul(h_prev, self.vars['wh']) # (batch_size, hidden_dim)
        users_vector = tf.matmul(x, self.adj) # (batch_size, number_of_users)
        users_vector = tf.matmul(users_vector, self.vars['wx']) # (batch_size, hidden_dim)

        output = tf.tanh(content_vector + users_vector) # (batch_size, hidden_dim)

        return output
