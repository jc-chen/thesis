from gcn.inits import *
from gcn.utils2 import tensor_diff
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# global unique layer ID dictionary for layer name assignment
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Helper function, assigns unique layer IDs."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    """Dropout for sparse tensors."""
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    """Wrapper for tf.matmul (sparse vs dense)."""
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)
    else:
        res = tf.matmul(x, y)
    return res


class Layer(object):
    """Base layer class. Defines basic API for all layer objects.
    Implementation inspired by keras (http://keras.io).

    # Properties
        name: String, defines the variable scope of the layer.
        logging: Boolean, switches Tensorflow histogram logging on/off

    # Methods
        _call(inputs): Defines computation graph of layer
            (i.e. takes input, returns output)
        __call__(inputs): Wrapper for _call()
        _log_vars(): Log all variables
    """

    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False
        self.molecule_partitions = None
        self.num_labels = 0

    def _call(self, inputs):
        return inputs

    def __call__(self, inputs):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs[-1])
            outputs = self._call(inputs)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])


class GraphConvolution(Layer):
    """Graph convolution layer."""
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            for i in range(len(self.support)):
                self.vars['weights_' + str(i)] = glorot([input_dim, output_dim],
                                                        name='weights_' + str(i))
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')

        if self.logging:
            self._log_vars()

    def _call(self, inputs):
        x = inputs[-1]

        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # convolve
        supports = list()
        for i in range(len(self.support)):
            if not self.featureless:
                pre_sup = dot(x, self.vars['weights_' + str(i)],
                              sparse=self.sparse_inputs)
            else:
                pre_sup = self.vars['weights_' + str(i)]
            support = dot(self.support[i], pre_sup, sparse=True)
            supports.append(support)
        output = tf.add_n(supports)

        # bias
        if self.bias:
            output += self.vars['bias']

        output = self.act(output)
        #output = tf.layers.batch_normalization(output)
        return output



class ReadOutSimple(Layer):
    def __init__(self, input_dim, features_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(ReadOutSimple, self).__init__(**kwargs)

        if dropout: self.dropout = placeholders['dropout']
        else: self.dropout = 0.
        self.act = act
        self.support = placeholders['support'] #support is the normalized adj matrix
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_labels = placeholders['labels'].shape[1]
        self.molecule_partitions = placeholders['molecule_partitions']
        self.num_molecules = placeholders['num_molecules']
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_i'] = glorot([input_dim, output_dim],
                                            name='weights_i')      
            if self.bias:
                self.vars['bias_i'] = zeros([output_dim],
                                            name='bias_i')
        if self.logging:
            self._log_vars()
        

    def _call(self, inputs):
        x = inputs[-1]
        
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            x = tf.sparse_tensor_to_dense(x,validate_indices=False)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        nn_i = tf.matmul(x,self.vars['weights_i'])
        
        if self.bias:
            nn_i += self.vars['bias_i']

        output = nn_i
        output = tf.cumsum(output)
        output = tf.gather(output,self.molecule_partitions)
        output = tensor_diff(self, output)

        return self.act(output)


class ReadOutSimpleAct(Layer):
    def __init__(self, input_dim, features_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(ReadOutSimpleAct, self).__init__(**kwargs)

        if dropout: self.dropout = placeholders['dropout']
        else: self.dropout = 0.
        self.act = act
        self.support = placeholders['support'] #support is the normalized adj matrix
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_labels = placeholders['labels'].shape[1]
        self.molecule_partitions = placeholders['molecule_partitions']
        self.num_molecules = placeholders['num_molecules']
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_i'] = glorot([input_dim, output_dim],
                                            name='weights_i')      
            if self.bias:
                self.vars['bias_i'] = zeros([output_dim],
                                            name='bias_i')
        if self.logging:
            self._log_vars()
        

    def _call(self, inputs):
        x = inputs[-1]
        
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            x = tf.sparse_tensor_to_dense(x,validate_indices=False)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        nn_i = tf.matmul(x,self.vars['weights_i'])
        
        if self.bias:
            nn_i += self.vars['bias_i']

        output = tf.nn.tanh(nn_i)
        output = tf.cumsum(output)
        output = tf.gather(output,self.molecule_partitions)
        output = tensor_diff(self, output)

        return self.act(output)


class ReadOutSimpleMulti(Layer):
    def __init__(self, input_dim, features_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(ReadOutSimpleMulti, self).__init__(**kwargs)

        if dropout: self.dropout = placeholders['dropout']
        else: self.dropout = 0.
        self.act = act
        self.support = placeholders['support'] #support is the normalized adj matrix
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_labels = placeholders['labels'].shape[1]
        self.molecule_partitions = placeholders['molecule_partitions']
        self.num_molecules = placeholders['num_molecules']
        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_i'] = glorot([input_dim, input_dim],
                                            name='weights_i')

            self.vars['weights_j'] = glorot([input_dim, input_dim],
                                            name='weights_j')   

            self.vars['weights_k'] = glorot([input_dim, output_dim],
                                            name='weights_k')         
            if self.bias:
                self.vars['bias_i'] = zeros([input_dim],
                                            name='bias_i')
                self.vars['bias_k'] = zeros([output_dim],
                                            name='bias_k')

        if self.logging:
            self._log_vars()
        

    def _call(self, inputs):
        x = inputs[-1]
        
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            x = tf.sparse_tensor_to_dense(x,validate_indices=False)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        nn_i = tf.matmul(x,self.vars['weights_i'])
        
        if self.bias:
            nn_i += self.vars['bias_i']

        nn_i = tf.matmul(nn_i,self.vars['weights_j'])
        nn_i = tf.nn.tanh(nn_i)
        nn_i = tf.matmul(nn_i,self.vars['weights_k'])

        if self.bias:
            nn_i += self.vars['bias_k']

        output = tf.nn.tanh(nn_i)
        output = tf.cumsum(output)
        output = tf.gather(output,self.molecule_partitions)
        output = tensor_diff(self, output)

        return self.act(output)



class ReadOutInitSingle(Layer):
    """from 'classes' weights to values"""
    def __init__(self, input_dim, features_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(ReadOutInitSingle, self).__init__(**kwargs)

        if dropout: self.dropout = placeholders['dropout']
        else: self.dropout = 0.

        self.act = act
        self.support = placeholders['support'] #support is the normalized adj matrix
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_labels = placeholders['labels'].shape[1]

        self.molecule_partitions = placeholders['molecule_partitions']
        self.num_molecules = placeholders['num_molecules']

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):

            self.vars['weights_i'] = glorot([input_dim, output_dim],
                                            name='weights_i')        
            self.vars['weights_j'] = glorot([features_dim, output_dim],
                                            name='weights_j')        
            if self.bias:
                self.vars['bias_i'] = zeros([output_dim],
                                            name='bias_i')
                self.vars['bias_j'] = zeros([output_dim],
                                            name='bias_j')          
        if self.logging:
            self._log_vars()
        
    def _call(self, inputs):
        x = inputs[-1]
        x_init = inputs[0]
        x_init =  sparse_dropout(x_init, 1-self.dropout, self.num_features_nonzero)
        x_init = tf.sparse_tensor_to_dense(x_init,validate_indices=False)
        
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            x = tf.sparse_tensor_to_dense(x,validate_indices=False)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        # b = tf.slice(x,[0,self.molecule_partitions[0],0],[1,self.molecule_partitions[1],x.shape[1]])
        #n_molecs = tf.scan(lambda a, x:a,)
        #tf.dynamic_partition(x, self.molecule_partitions, self.num_molecules, name=None)

        nn_i = tf.matmul(x,self.vars['weights_i'])
        nn_j = tf.matmul(x_init,self.vars['weights_j'])
        
        if self.bias:
            nn_i += self.vars['bias_i']
            nn_j += self.vars['bias_j']


        nn_i = tf.nn.sigmoid(nn_i)
        nn_j = tf.nn.tanh(nn_j)

        output = tf.multiply(nn_i,nn_j)
        output = tf.cumsum(output)
        output = tf.gather(output,self.molecule_partitions)

        output = tensor_diff(self, output)
        
        #     if not self.featureless:
        #         pre_sup = dot(x, self.vars['weights_' + str(i)],
        #                       sparse=self.sparse_inputs)
        #     else:
        #         pre_sup = self.vars['weights_' + str(i)]
        #     support = dot(self.support[i], pre_sup, sparse=True)
        #     supports.append(support)
        # output = tf.add_n(supports)

        return self.act(output)


class ReadOutInitMulti(Layer):
    """from 'classes' weights to values"""
    def __init__(self, input_dim, features_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(ReadOutInitMulti, self).__init__(**kwargs)

        if dropout: self.dropout = placeholders['dropout']
        else: self.dropout = 0.

        self.act = act
        self.support = placeholders['support'] #support is the normalized adj matrix
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_labels = placeholders['labels'].shape[1]

        self.molecule_partitions = placeholders['molecule_partitions']
        self.num_molecules = placeholders['num_molecules']

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):

            self.vars['weights_i'] = glorot([input_dim, input_dim],
                                            name='weights_i')        
            self.vars['weights_j'] = glorot([features_dim, features_dim],
                                            name='weights_j')        
            self.vars['weights_ik'] = glorot([input_dim, input_dim],
                                            name='weights_ik')        
            self.vars['weights_jl'] = glorot([features_dim, input_dim],
                                            name='weights_jl')          
            self.vars['weights_im'] = glorot([input_dim, output_dim],
                                            name='weights_im')            
            self.vars['weights_jn'] = glorot([input_dim, output_dim],
                                            name='weights_jn')             
            if self.bias:
                self.vars['bias_i'] = zeros([output_dim],
                                            name='bias_i')
                self.vars['bias_j'] = zeros([output_dim],
                                            name='bias_j')          
        if self.logging:
            self._log_vars()
        
    def _call(self, inputs):
        x = inputs[-1]
        x_init = inputs[0]
        x_init =  sparse_dropout(x_init, 1-self.dropout, self.num_features_nonzero)
        x_init = tf.sparse_tensor_to_dense(x_init,validate_indices=False)
        
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            x = tf.sparse_tensor_to_dense(x,validate_indices=False)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        nn_i = tf.matmul(x,self.vars['weights_i'])
        nn_j = tf.matmul(x_init,self.vars['weights_j'])

        nn_i = tf.matmul(nn_i,self.vars['weights_ik'])        
        nn_i = tf.nn.tanh(nn_i)
        nn_i = tf.matmul(nn_i,self.vars['weights_im'])

        nn_j = tf.matmul(nn_j,self.vars['weights_jl'])
        nn_j = tf.nn.leaky_relu(nn_j)
        nn_j = tf.matmul(nn_j,self.vars['weights_jn'])
        
        if self.bias:
            nn_i += self.vars['bias_i']
            nn_j += self.vars['bias_j']

        nn_i = tf.nn.sigmoid(nn_i)
        nn_j = tf.nn.tanh(nn_j)

        output = tf.multiply(nn_i,nn_j)
        output = tf.cumsum(output)
        output = tf.gather(output,self.molecule_partitions)
        output = tensor_diff(self, output)

        return self.act(output)



class ReadOutInitMultiII(Layer):
    def __init__(self, input_dim, features_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(ReadOutInitMultiII, self).__init__(**kwargs)

        if dropout: self.dropout = placeholders['dropout']
        else: self.dropout = 0.

        self.act = act
        self.support = placeholders['support'] #support is the normalized adj matrix
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_labels = placeholders['labels'].shape[1]
        self.molecule_partitions = placeholders['molecule_partitions']
        self.num_molecules = placeholders['num_molecules']

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_i'] = glorot([input_dim, features_dim],
                                            name='weights_i')        
            self.vars['weights_j'] = glorot([features_dim, output_dim],
                                            name='weights_j')        
            self.vars['weights_k'] = glorot([features_dim, output_dim],
                                            name='weights_k')       
            if self.bias:
                self.vars['bias_i'] = zeros([output_dim],
                                            name='bias_i')
                self.vars['bias_j'] = zeros([output_dim],
                                            name='bias_j')
        if self.logging:
            self._log_vars()
        

    def _call(self, inputs):
        x = inputs[-1]
        x_init = inputs[0]
        x_init =  sparse_dropout(x_init, 1-self.dropout, self.num_features_nonzero)
        x_init = tf.sparse_tensor_to_dense(x_init,validate_indices=False)
        
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            x = tf.sparse_tensor_to_dense(x,validate_indices=False)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        nn_i = tf.matmul(x,self.vars['weights_i'])
        nn_i = tf.multiply(nn_i,x_init)
        nn_i = tf.matmul(nn_i,self.vars['weights_k'])
        nn_j = tf.matmul(x_init,self.vars['weights_j'])
        
        if self.bias:
            nn_i += self.vars['bias_i']
            nn_j += self.vars['bias_j']

        nn_i = tf.nn.sigmoid(nn_i)
        nn_j = tf.nn.tanh(nn_j)

        output = tf.multiply(nn_i,nn_j)
        output = tf.layers.batch_normalization(output)
        output = tf.cumsum(output)
        output = tf.gather(output,self.molecule_partitions)
        output = tensor_diff(self, output)
        return self.act(output)



class ReadOut4(Layer):
    def __init__(self, input_dim, features_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, **kwargs):
        super(ReadOut4, self).__init__(**kwargs)

        if dropout: self.dropout = placeholders['dropout']
        else: self.dropout = 0.

        self.act = act
        self.support = placeholders['support'] #support is the normalized adj matrix
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_labels = placeholders['labels'].shape[1]
        self.molecule_partitions = placeholders['molecule_partitions']
        self.num_molecules = placeholders['num_molecules']

        # helper variable for sparse dropout
        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_i'] = glorot([input_dim, features_dim],
                                            name='weights_i')        
            self.vars['weights_j'] = glorot([features_dim, output_dim],
                                            name='weights_j')        
            self.vars['weights_k'] = glorot([features_dim, output_dim],
                                            name='weights_k')       
            if self.bias:
                self.vars['bias_i'] = zeros([output_dim],
                                            name='bias_i')
                self.vars['bias_j'] = zeros([output_dim],
                                            name='bias_j')
        if self.logging:
            self._log_vars()
        

    def _call(self, inputs):
        x = inputs[-1]
        x_init = inputs[0]
        x_init =  sparse_dropout(x_init, 1-self.dropout, self.num_features_nonzero)
        x_init = tf.sparse_tensor_to_dense(x_init,validate_indices=False)
        
        # dropout
        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            x = tf.sparse_tensor_to_dense(x,validate_indices=False)
        else:
            x = tf.nn.dropout(x, 1-self.dropout)

        nn_i = tf.matmul(x,self.vars['weights_i'])
        nn_i = tf.multiply(nn_i,x_init)
        nn_i = tf.matmul(nn_i,self.vars['weights_k'])
        nn_j = tf.matmul(x_init,self.vars['weights_j'])
        
        if self.bias:
            nn_i += self.vars['bias_i']
            nn_j += self.vars['bias_j']

        nn_i = tf.nn.sigmoid(nn_i)
        nn_j = tf.nn.leaky_relu(nn_j)

        output = tf.multiply(nn_i,nn_j)
        output = tf.cumsum(output)
        output = tf.gather(output,self.molecule_partitions)
        output = tensor_diff(self, output)

        return self.act(output)