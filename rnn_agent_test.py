import rnn_speaker_agent as sa
import rnn_listener_agent as la

from shared_lib import vocabulary, utils

import copy
import numpy as np
import tensorflow as tf

import unittest


class TestRNNAgentCore(unittest.TestCase):
    def setUp(self):
        self.g = tf.Graph()
        model_params = dict(corpus_vocab_size=512, embedding_size=100, agent_vocab_size=50, num_layers=1, graph=self.g)

        self.speaker = sa.RNNSpeakerAgent(**model_params)
        self.speaker.BuildCoreGraph()
        self.speaker.BuildTrainGraph()
        self.speaker.BuildOutputGraph()

        model_params['message_tensor'] = self.speaker.message_
        self.listener = la.RNNListenerAgent(**model_params)
        self.listener.BuildCoreGraph()
        self.listener.BuildTrainGraph()
        self.listener.BuildOutputGraph()

    def test_shapes_embed(self):
        self.assertEqual(self.speaker.W_in_.get_shape().as_list(), [512, 100])

    def test_shapes_recurrent(self):
        self.assertEqual(self.speaker.cell_.state_size[0].c, 100)
        self.assertEqual(self.speaker.cell_.state_size[0].h, 100)
        init_c_shape = self.speaker.initial_h_[0].c.get_shape().as_list()
        init_h_shape = self.speaker.initial_h_[0].h.get_shape().as_list()
        self.assertEqual(init_c_shape, [None, 100])
        self.assertEqual(init_h_shape, [None, 100])

        self.assertEqual(self.speaker.final_h_[0].c.get_shape().as_list(),
                         init_c_shape)
        self.assertEqual(self.speaker.final_h_[0].h.get_shape().as_list(),
                         init_h_shape)

    def test_shapes_output(self):
        self.assertEqual(self.speaker.W_out_.get_shape().as_list(), [100, 512])
        self.assertEqual(self.speaker.b_out_.get_shape().as_list(), [512])
        self.assertEqual(self.speaker.loss_.get_shape().as_list(), [])


class TestRNNAgentTrain(unittest.TestCase):
    def setUp(self):
        model_params = dict(V=512, H=100, num_layers=1)
        self.speaker = rnnlm.RNNLM(**model_params)
        self.speaker.BuildCoreGraph()
        self.speaker.BuildTrainGraph()

    def test_shapes_train(self):
        self.assertEqual(self.speaker.train_loss_.get_shape().as_list(), [])
        self.assertNotEqual(self.speaker.loss_, self.speaker.train_loss_)
        self.assertIsNotNone(self.speaker.train_step_)


#class TestRNNLMSampler(unittest.TestCase):
#    def setUp(self):
#        model_params = dict(V=512, H=100, num_layers=1)
#        self.speaker = rnnlm.RNNLM(**model_params)
#        self.speaker.BuildCoreGraph()
#        self.speaker.BuildSamplerGraph()
#
#    def test_shapes_sample(self):
#        self.assertEqual(self.speaker.pred_samples_.get_shape().as_list(),
#                         [None, None, 1])

class RunEpochTester(unittest.TestCase):
    def setUp(self):
        sequence = ["a", "b", "c", "d"]
        self.vocab = vocabulary.Vocabulary(sequence)
        ids = self.vocab.words_to_ids(sequence)
        self.train_ids = np.array(ids * 10000, dtype=int)
        self.test_ids = np.array(ids * 100, dtype=int)

        model_params = dict(V=self.vocab.size, H=10,
                            softmax_ns=2, num_layers=1)
        self.speaker = rnnlm.RNNLM(**model_params)
        self.speaker.BuildCoreGraph()
        self.speaker.BuildTrainGraph()
        self.speaker.BuildSamplerGraph()
        # For toy model, ignore sampled softmax.
        self.speaker.train_loss_ = self.speaker.loss_

    def injectCode(self, run_epoch_fn, score_dataset_fn):
        self.run_epoch = run_epoch_fn
        # TODO: fix return value in notebook
        # self.score_dataset = score_dataset_fn
        # TODO: remove this.
        def _score_dataset(lm, session, ids, name="Data"):
            # For scoring, we can use larger batches to speed things up.
            bi = utils.batch_generator(ids, batch_size=100, max_time=100)
            cost = self.run_epoch(lm, session, bi, 
                             learning_rate=1.0, train=False, 
                             verbose=False, tick_s=3600)
            print "%s: avg. loss: %.03f  (perplexity: %.02f)" % (name, cost, np.exp(cost))
            return cost
        self.score_dataset = _score_dataset

    def test_toy_model(self):
        with tf.Session(graph=self.speaker.graph) as sess:
            sess.run(tf.global_variables_initializer())
            bi = utils.batch_generator(self.train_ids, 5, 10)
            self.run_epoch(self.speaker, sess, bi, learning_rate=0.5,
                           train=True, verbose=True, tick_s=1.0)
            train_loss = self.score_dataset(self.speaker, sess, self.train_ids,
                                            name="Train set")
            test_loss = self.score_dataset(self.speaker, sess, self.test_ids,
                                           name="Test set")
        # This is a *really* simple dataset, so you should have no trouble
        # getting almost perfect scores.
        self.assertFalse(train_loss is None)
        self.assertFalse(test_loss is None)
        self.assertLessEqual(train_loss, 0.05)
        self.assertLessEqual(test_loss, 0.05)

