import time

import tensorflow as tf
import numpy as np


def matmul3d(X, W):
    """Wrapper for tf.matmul to handle a 3D input tensor X.
    Will perform multiplication along the last dimension.

    Args:
      X: [m,n,k]
      W: [k,l]

    Returns:
      XW: [m,n,l]
    """
    Xr = tf.reshape(X, [-1, tf.shape(X)[2]])
    XWr = tf.matmul(Xr, W)
    newshape = [tf.shape(X)[0], tf.shape(X)[1], tf.shape(W)[1]]
    return tf.reshape(XWr, newshape)


def MakeFancyRNNCell(H, keep_prob, num_layers=1, name_scope="listener"):
    """Make a fancy RNN cell.

    Use tf.nn.rnn_cell functions to construct an LSTM cell.
    Initialize forget_bias=0.0 for better training.

    Args:
      H: hidden state size
      keep_prob: dropout keep prob (same for input and output)
      num_layers: number of cell layers

    Returns:
      (tf.nn.rnn_cell.RNNCell) multi-layer LSTM cell with dropout
    """
    with tf.variable_scope("listener"):
        with tf.variable_scope(name_scope):
            cells = []
            for _ in xrange(num_layers):
              cell = tf.contrib.rnn.BasicLSTMCell(H, forget_bias=0.0)
              cell = tf.contrib.rnn.DropoutWrapper(
                  cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)
              cells.append(cell)
            return tf.contrib.rnn.MultiRNNCell(cells)


# Decorator-foo to avoid indentation hell.
# Decorating a function as:
# @with_self_graph
# def foo(self, ...):
#     # do tensorflow stuff
#
# Makes it behave as if it were written:
# def foo(self, ...):
#     with self.graph.as_default():
#         # do tensorflow stuff
#
# We hope this will save you some indentation, and make things a bit less
# error-prone.
def with_self_graph(function):
    def wrapper(self, *args, **kwargs):
        with self.graph.as_default():
            return function(self, *args, **kwargs)
    return wrapper


class RNNListenerAgent(object):
    def __init__(self, graph=None, *args, **kwargs):
        """Init function.

        This function just stores hyperparameters. You'll do all the real graph
        construction in the Build*Graph() functions below.

        Args:
          V: vocabulary size
          H: hidden state dimension
          num_layers: number of RNN layers (see tf.nn.rnn_cell.MultiRNNCell)
        """
        # Set TensorFlow graph. All TF code will work on this graph.
        self.graph = graph or tf.Graph()
        self.SetParams(*args, **kwargs)

    @with_self_graph
    def SetParams(self, corpus_vocab_size, embedding_size, agent_vocab_size, message_tensor, 
      softmax_ns=200, num_layers=1):
        # Model structure; these need to be fixed for a given model.
        self.corpus_vocab = corpus_vocab_size
        self.embedding = embedding_size
        self.agent_vocab = agent_vocab_size
        self.num_layers = num_layers

        # Training hyperparameters; these can be changed with feed_dict,
        # and you may want to do so during training.
        with tf.name_scope("Training_Parameters"):
            # Number of samples for sampled softmax.
            self.softmax_ns = softmax_ns

            self.learning_rate_ = tf.placeholder(tf.float32, [], name="learning_rate")

            # For gradient clipping, if you use it.
            # Due to a bug in TensorFlow, this needs to be an ordinary python
            # constant instead of a tf.constant.
            self.max_grad_norm_ = 5.0

            self.use_dropout_ = tf.placeholder_with_default(
                False, [], name="use_dropout")

            # Message vector.  Should be a one-hot vector of size agent_vocab.
            self.input_m_ = message_tensor

            # If use_dropout is fed as 'True', this will have value 0.5.
            self.dropout_keep_prob_ = tf.cond(
                self.use_dropout_,
                lambda: tf.constant(0.5),
                lambda: tf.constant(1.0),
                name="dropout_keep_prob")

            # Dummy for use later.
            self.no_op_ = tf.no_op()


    @with_self_graph
    def BuildCoreGraph(self):
        """Construct the core RNNLM graph, needed for any use of the model.

        This should include:
        - Placeholders for input tensors (input_w_, initial_h_, target_y_)
        - Variables for model parameters
        - Tensors representing various intermediate states
        - A Tensor for the final state (final_h_)
        - A Tensor for the output logits (logits_), i.e. the un-normalized argument
          of the softmax(...) function in the output layer.
        - A scalar loss function (loss_)

        Your loss function should be a *scalar* value that represents the
        _average_ loss across all examples in the batch (i.e. use tf.reduce_mean,
        not tf.reduce_sum).

        You shouldn't include training or sampling functions here; you'll do
        this in BuildTrainGraph and BuildSampleGraph below.

        We give you some starter definitions for input_w_ and target_y_, as
        well as a few other tensors that might help. We've also added dummy
        values for initial_h_, logits_, and loss_ - you should re-define these
        in your code as the appropriate tensors.

        See the in-line comments for more detail.
        """
        with tf.variable_scope("listener"):
            # Input ids, with dynamic shape depending on input.
            # Should be shape [batch_size, max_time] and contain integer word indices.
            self.input_w1_ = tf.placeholder(tf.int32, [None, None], name="w1")

            # Input ids, with dynamic shape depending on input.
            # Should be shape [batch_size, max_time] and contain integer word indices.
            self.input_w2_ = tf.placeholder(tf.int32, [None, None], name="w2")

            # Initial hidden state. You'll need to overwrite this with cell.zero_state
            # once you construct your RNN cell.
            self.initial_h_ = None

            # Final hidden state. You'll need to overwrite this with the output from
            # tf.nn.dynamic_rnn so that you can pass it in to the next batch (if
            # applicable).
            self.final_h_ = None

            # Output logits, which can be used by loss functions or for prediction.
            # Overwrite this with an actual Tensor of shape
            # [batch_size, max_time, V].
            self.logits_ = None

            # Should be the same shape as inputs_w_
            self.target_y_ = tf.placeholder(tf.int32, [None, None], name="y")

            # Replace this with an actual loss function
            self.loss_ = None

            # Get dynamic shape info from inputs
            with tf.name_scope("batch_size"):
                self.batch_size_ = tf.shape(self.input_w1_)[0]
            with tf.name_scope("max_time"):
                self.max_time_ = tf.shape(self.input_w1_)[1]

            # Get sequence length from input_w_.
            # TL;DR: pass this to dynamic_rnn.
            # This will be a vector with elements ns[i] = len(input_w_[i])
            # You can override this in feed_dict if you want to have different-length
            # sequences in the same batch, although you shouldn't need to for this
            # assignment.
            self.ns_ = tf.tile([self.max_time_], [self.batch_size_, ], name="ns")

            #### YOUR CODE HERE ####
            # See hints in instructions!

            # Construct embedding layer
            with tf.name_scope("Listener_Embedding_Layer"):
                self.W_in_ = tf.Variable(tf.random_uniform([self.corpus_vocab, self.embedding], 0.0, 1.0), name="W_in_")
                self.x1_ = tf.reshape(tf.nn.embedding_lookup(self.W_in_, self.input_w1_), 
                                      [self.batch_size_, self.max_time_, self.embedding], name = "x1_")
                self.x2_ = tf.reshape(tf.nn.embedding_lookup(self.W_in_, self.input_w2_), 
                                      [self.batch_size_, self.max_time_, self.embedding], name = "x2_")
                self.x_ = tf.concat([self.x1_, self.x2_, 
                                     tf.reshape(self.input_m_, [self.batch_size_, self.max_time_, self.agent_vocab])], 
                                    2, name = "x_")

            # Construct RNN/LSTM cell and recurrent layer.
            with tf.name_scope("Listener_Recurrent_Layer"):
                self.cell_ = MakeFancyRNNCell(2*self.embedding+self.agent_vocab, self.dropout_keep_prob_, self.num_layers)

                self.initial_h_ = self.cell_.zero_state(self.batch_size_, tf.float32)
                self.o_, self.final_h_ = tf.nn.dynamic_rnn(self.cell_, self.x_, initial_state=self.initial_h_, 
                                                           sequence_length=self.ns_)
                #self.o_, self.final_h_ = tf.nn.dynamic_rnn(self.cell_, self.x_, initial_state=self.initial_h_)


            # Softmax output layer, over vocabulary. Just compute logits_ here.
            # Hint: the matmul3d function will be useful here; it's a drop-in
            # replacement for tf.matmul that will handle the "time" dimension
            # properly.
            with tf.name_scope("Listener_Output_Layer"):
                self.W_out_ = tf.Variable(tf.random_uniform([2*self.embedding+self.agent_vocab, 1], 0.0, 1.0), name="W_out_")
                self.b_out_ = tf.Variable(tf.zeros([1]), name="b_out_")
                self.logits_ = matmul3d(self.o_, self.W_out_) + self.b_out_



            # Loss computation (true loss, for prediction)
            self.loss_ = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.logits_, labels=self.target_y_), name = "loss_")


            #### END(YOUR CODE) ####

    @with_self_graph
    def BuildTrainGraph(self):
        """Construct the training ops.

        You should define:
        - train_loss_ : sampled softmax loss, for training
        - train_step_ : a training op that can be called once per batch

        Your loss function should be a *scalar* value that represents the
        _average_ loss across all examples in the batch (i.e. use tf.reduce_mean,
        not tf.reduce_sum).
        """
        with tf.variable_scope("listener"):
            # Replace this with an actual training op
            self.train_step_ = None

            # Replace this with an actual loss function
            self.train_loss_ = None

            #### YOUR CODE HERE ####
            # See hints in instructions!

            # Define approximate loss function.
            # Note: self.softmax_ns (i.e. k=200) is already defined; use that as the
            # number of samples.
            # Loss computation (sampled, for training)
            self.train_loss_ = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights=tf.transpose(self.W_out_), biases=self.b_out_, 
                                           inputs=tf.reshape(self.o_, [-1, 2*self.embedding+self.agent_vocab]), 
                                           labels=tf.reshape(self.target_y_, [-1, 1]), 
                                           num_sampled=self.softmax_ns, num_classes=1))


            # Define optimizer and training op
            self.train_step_ = tf.train.AdagradOptimizer(self.learning_rate_).minimize(self.train_loss_)


            #### END(YOUR CODE) ####

#    @with_self_graph
#    def BuildSamplerGraph(self):
#        """Construct the sampling ops.
#
#        You should define pred_samples_ to be a Tensor of integer indices for
#        sampled predictions for each batch element, at each timestep.
#
#        Hint: use tf.multinomial, along with a couple of calls to tf.reshape
#        """
#        # Replace with a Tensor of shape [batch_size, max_time, num_samples = 1]
#        self.pred_samples_ = None
#
#        #### YOUR CODE HERE ####
#        self.pred_samples_ = tf.reshape(
#            tf.multinomial(tf.reshape(self.logits_, [-1, 1]), 1),
#            [ self.batch_size_, self.max_time_, 1 ])
#
#
#
#        #### END(YOUR CODE) ####

    @with_self_graph
    def BuildOutputGraph(self):
        """ Construct the output graph for the listener agent.
        """
        with tf.variable_scope("listener"):
            self.prediction_ = tf.nn.softmax(self.logits_, name = "prediction_")

