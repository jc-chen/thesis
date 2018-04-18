   ONE LAYER
        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.node_output_size,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout=False,
                                            sparse_inputs=True,
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

        3 layers

         self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout=False,
                                            sparse_inputs=True,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=FLAGS.hidden6,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout=False,
                                            bias=True,
                                            logging=self.logging))


        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden6,
                                            output_dim=FLAGS.node_output_size,
                                            placeholders=self.placeholders,
                                            act=tf.nn.tanh,
                                            bias=True,
                                            dropout=False,


5 layers


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
                                            output_dim=FLAGS.hidden6,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout = False,
                                            bias=False,
                                            logging=self.logging))


        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden6,
                                            output_dim=FLAGS.node_output_size,
                                            placeholders=self.placeholders,
                                            act=tf.nn.tanh,
                                            bias=True,
                                            dropout=False,




    9 layers

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
                                            output_dim=FLAGS.hidden6,
                                            placeholders=self.placeholders,
                                            act=tf.nn.leaky_relu,
                                            dropout = False,
                                            bias=False,
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




