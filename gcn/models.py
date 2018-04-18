from gcn.layers import *
from gcn.metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.normalized_targets = None
        self.target_masks = None

        self.molecule_partitions = None
        self.num_molecules = None

        self.mae = 0
        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

        self.target_mean = None
        self.target_stdev = None

        self.print_me = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        #print(self.inputs.get_shape(),"eeeeeeee")

        with tf.variable_scope(self.name):
            self._build()
            self.target_mean = tf.Variable(tf.zeros([self.placeholders['labels'].shape[1].value]), dtype=tf.float32, trainable=False)
            self.target_stdev = tf.Variable(tf.zeros([self.placeholders['labels'].shape[1].value]), dtype=tf.float32, trainable=False)

        #Setting for easy access
        self.target_mask = self.placeholders['labels_mask']

        #For normalizing data
        self.normalized_targets = (self.placeholders['labels']-self.target_mean)/self.target_stdev

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer([self.activations[0], self.activations[-1]])
            self.activations.append(hidden)
            #self.layers.append(tf.layers.batch_normalization(self.activations[-1]))
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}
        
        #For getting normalization parameters from training data
        [norm_mean,norm_std]=self.get_normalize_parameters()
        self.get_normalize_op = tf.group(tf.assign(self.target_mean,norm_mean),tf.assign(self.target_stdev,norm_std))
        self.get_mean = tf.assign(self.target_mean,norm_mean)
        self.get_std = tf.assign(self.target_stdev,norm_std)

        # Build metrics
        self._loss()
        self._accuracy()
        self._mae()

        #Optimization op
        self.opt_op = self.optimizer.minimize(self.loss)

    def get_mae(self):
        return self.mae

    def predict(self):
        pass

    def get_normalize_parameters(self):
        labels = self.placeholders['labels']
        mask = tf.cast(self.placeholders['labels_mask'], dtype=tf.float32)

        num_items = tf.reduce_sum(mask)

        #mask fixes
        mask = tf.expand_dims(mask,-1)
        mask = tf.tile(mask,[1,labels.shape[1].value])

        mean = tf.reduce_sum(labels*mask,axis=0)/num_items

        std = 2.0*tf.sqrt(tf.reduce_sum(tf.square(labels-mean)*mask,axis=0)/(num_items-1))

        return mean,std
        
    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError


    def _mae(self):
        raise NotImplementedError

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



class JCNN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(JCNN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions

        # this should be molecule outputs now
        self.molecule_number_of_outputs = placeholders['labels'].get_shape().as_list()[1]
        self.molecule_partitions = placeholders['molecule_partitions']
        self.num_molecules = placeholders['num_molecules']
        self.placeholders = placeholders
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss (regularization) --uncomment to reduce overfitting
        #for var in self.layers[0].vars.values():
        #    self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # L2 loss
        self.loss += square_error(self.outputs, self.normalized_targets,self.target_mask)

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.normalized_targets,
                                        self.target_mask,self.target_mean,
                                        self.target_stdev)
    def _mae(self):
        self.mae = mean_absolute_error(self.outputs, self.normalized_targets,
                                    self.target_mask)
    # def _maeUnnormalized(self):
    #     self.mae = mean_absolute_error_unnormalized(self.outputs, self.normalized_targets,
    #                                 self.target_mask)

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout=False,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden2,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout=False,
                                            bias=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden2,
                                            output_dim=FLAGS.hidden3,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout = False,
                                            bias=False,
                                            logging=self.logging))


        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden3,
                                            output_dim=FLAGS.hidden4,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout = False,
                                            bias=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden4,
                                            output_dim=FLAGS.hidden5,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout = False,
                                            bias=False,
                                            logging=self.logging))


        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden5,
                                            output_dim=FLAGS.hidden6,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout = False,
                                            bias=True,
                                            logging=self.logging))


        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden6,
                                            output_dim=FLAGS.node_output_size,
                                            placeholders=self.placeholders,
                                            act=tf.nn.tanh,
                                            bias=True,
                                            dropout=False,
                                            logging=self.logging))
        
        self.layers.append(ReadOutSimpleAct(input_dim=FLAGS.node_output_size, 
                                    features_dim = self.input_dim,
                                    output_dim=self.molecule_number_of_outputs,
                                    placeholders=self.placeholders,
                                    act=lambda x: x,
                                    dropout=False,
                                    bias=True,
                                    sparse_inputs=False,
                                    logging=self.logging))
    def predict(self):
        return self.outputs*self.target_stdev+self.target_mean