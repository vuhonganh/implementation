import tensorflow as tf



class Word2Vec:
    def __init__(self, vocab_size, embed_size, num_sampled, learning_rate):
        self.vs = vocab_size
        self.es = embed_size
        self.num_sampled = num_sampled
        self.lr = learning_rate

    def _create_placeholder(self):
        self.input_placeholder = tf.placeholder(tf.int32, shape=None, name='input')
        # label_placeholder has to have shape (batch_size, 1) because tf.nn.nce_loss requires that
        self.label_placeholder = tf.placeholder(tf.int32, shape=(None, 1), name='label')

    def _create_weights(self):
        # embeddings matrix (which is equivalent to input matrix V in my note)
        self.embeddings = tf.Variable(tf.random_uniform(shape=(self.vs, self.es), minval=-1.0, maxval=1.0),
                                      name='embeddings')
    def _create_loss(self):
        # embedding of the current this batch
        embed = tf.nn.embedding_lookup(self.embeddings, self.input_placeholder)
        # nce_weight matrix, which is equal to output matrix U in my note
        nce_weight = tf.Variable(tf.random_normal(shape=(self.vs, self.es),
                                                  mean=0.0,
                                                  stddev=1.0/(self.es ** 0.5)),
                                 name='nce_weight')
        nce_bias = tf.Variable(tf.zeros(self.vs), name='nce_bias')
        loss_ar = tf.nn.nce_loss(nce_weight, nce_bias, self.label_placeholder,
                                 embed, self.num_sampled, self.vs)
        self.loss = tf.reduce_mean(loss_ar)

    def _create_optimizer(self):
        self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def _create_summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.histogram('histogram loss', self.loss)
            # merge all summaries make it easier to manage
            self.summary_op = tf.summary.merge_all()

    def build_model(self):
        self._create_placeholder()
        self._create_weights()
        self._create_loss()
        self._create_optimizer()
        self._create_summary()
