# Prediction of Molecular Properties with Graph Convolution-based Neural Networks

This is a short guide on how to train and run the neural network. For more details about the project, please refer to my thesis write-up under writeup.pdf. The code in this repository is based upon the work of Kipf & Welling [(github.com/tkipf/gcn)](github.com/tkipf/gcn). The molecular data used to train the network is from the QM9 Dataset [(quantum-machine.org/datasets)](quantum-machine.org/datasets).



### Building a neural network using layers

#### Layer class

TODO -structure of a neural network: a sequence of layers; size/type of each layer, how many layers you have, how you order the layers


#### Model class
TODO
To adjust the sequence of layers......
_build()
append layers (Convolution, readout)


### Getting data
Raw molecular data from the QM9 Dataset require some pre-processing so that python can understand some of the floats in the text files. Run `python process_data.py <data directory> <output directory>` to process the entire batch. To randomly select n molecules and copy them to a new directory for training/testing, use randomize_data.py: 
```python randomize_data.py <source directory> <destination directory> <n>```



### Running train.py
Simply run ```python train.py <data directory> ``` to train, validate, and test on molecules saved in \<data directory\>. There are various flags you can set via ```python train.py <data directory> --optional_flag <flag_value>``` discussed below. 



#### Parsing data, saving and loading parsed data
`load_data(...)` defined in utils.py handles parsing the .xyz files, putting relevant information into matrices, normalizing features, and separating into training/validation/test sets. To load data for testing only, use `load_test_data()` instead. To save these matrices (and you will want to save them--for repeatability and efficiency, since building them takes some time), set the flag `data_output_path` to whichever directory you want to save in. i.e.
```
python train.py <data directory> --data_output_path <output directory>
```


To load the matrices, run train.py with the directory in which you saved the matrices, and set the `should_load_previous_data` flag to `True`, i.e. 

```
python train.py <data directory> --should_load_previous_data True
```


#### Saving and loading models
To save a model after training, use the flag `output_name` defined in train.py; to load a previously trained model (e.g. to test or further train the model), use the flag `input_name`. Models are, by default, saved to the `models/` directory, but that can be easily changed with the `dir_model` flag. In addition to saving the model itself, tensorboard data will be saved under tensorboard/<model_name> . Note if `output_name` is unspecified in a training session, the model will automatically be saved in Models/unnamed_model/

As an example, to load model A in a session and save it as model B when the session ends, run
```
python train.py <data directory> --input_name model_A --output_name model_B 
```

Note that this is independent from saving/loading parsed data. If you also want to save the parsed data, run
```
python train.py <data directory> --input_name model_A --output_name model_B --data_output_path <output dir>
```


#### Variable flags
There are several values that need to be tweaked, depending on what kind of training session you are running. Play around with these values and see how they affect your model! These are:

`learning_rate`: Defines the magnitude of change in your first training step/epoch. Since we are using AdamOptimizer to adjust learning rate during training, the exact value of the initial learning rate does not play a strong role as long as it is not absurdly large/small. 

`epochs`: The max number of epochs to train. 

`hiddenx`: The number of nodes in hidden layer x. 

`node_output_size`: The number of hidden features each node has prior to the readout phase. 

`dropout`: The probability of a node's value being ignored (dropped out), 0<=`dropout`<1. Dropout reduces overfitting, but if set to be too large, the neural network may not have enough information to train on. 

`early_stopping`: The min number of epochs to train. 


#### Testing
To use the neural network for prediction, run results.py  TODO

### Visualization with tensorboard
To open tensorboard, run `tensorboard --logdir <path_to_this_project>/tensorboard/` in the terminal and navigate to localhost:6006 in a web browser. In the bottom left of the page you can tick and untick particular models you have saved to compare them.

It is currently sent to generate tensorboard data only at the end of the session. However, if you are running a long training session and want the visualization to update live as you go, move the `visualization_writer.flush()` statement directly under `visualization_writer.add_summary(summary, epoch)` inside the `for epoch` loop. As flush is a costly operation, only let it execute every N epochs. 


To add variables you want to track, modify the `initialize_tensorboard_outputs()` function in train.py to include them. E.g. 

```
def initialize_tensorboard_outputs():
    train_loss = tf.Variable(0.)
    train_acc = tf.Variable(0.)
    variable_to_track = tf.Variable(initial value)

    a = tf.summary.scalar("Train Loss", train_loss)
    b = tf.summary.scalar("Train Acc", train_acc)
    c = tf.summary.scalar("variable", variable_to_track)

    summary_ops = tf.summary.merge([a, b, c])

    visualization_data = {
      'train_loss': train_loss,
      'train_accuracy': train_acc,
      'tracked_variable': variable_to_track,
    }
    return visualization_data, summary_ops

```

