import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

class TFActor:
    """Actor (Policy) Model."""

    def __init__(self, session, state_size, action_size, name, checkpoint_file=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        self.sess = session
        self.name = name

        self.action_size = action_size

        if checkpoint_file is None:
            with tf.variable_scope("actor_placeholders_"+self.name):
                self.input = tf.placeholder(tf.float32, shape=(None, state_size), name='input')
                # self.critic_input = tf.placeholder(tf.Tensor, shape=(None, 1), name='critic_input')

            self.output = self._inference()
            self.loss, self.optimizer = self._training_graph()

        else:
            checkpoint_dir = '/'.join(checkpoint_file.split('/')[:-1])
            saver = tf.train.import_meta_graph(checkpoint_file+'.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))

            self.input = tf.get_default_graph().get_tensor_by_name('actor_placeholders_'+self.name+'/input:0')
            # self.critic_input = tf.get_default_graph().get_tensor_by_name('actor_placeholders_'+self.name+'/critic_input:0')
            self.loss = tf.get_default_graph().get_tensor_by_name(f'actor_training_{self.name}/loss:0')
            self.optimizer = tf.get_default_graph().get_operation_by_name(f'actor_training_{self.name}/optimize')
            self.output = tf.get_default_graph().get_tensor_by_name(f'actor_inference_{self.name}/dense_2/BiasAdd:0')

        self.step = 0

    def _inference(self):
        with tf.variable_scope("actor_inference_"+self.name):
            layer = tf.layers.batch_normalization(self.input)
            layer = tf.layers.dense(layer, 300, activation=tf.nn.relu)
            layer = tf.layers.batch_normalization(layer)
            layer = tf.layers.dense(layer, 200, activation=tf.nn.relu)
            layer = tf.layers.batch_normalization(layer)
            output = tf.layers.dense(layer, self.action_size, activation=tf.nn.tanh)
        return output

    def _training_graph(self):
        with tf.variable_scope('actor_training_'+self.name):
            # loss = -self.critic_input
            # loss = tf.reduce_mean(loss, name='loss')
            loss = tf.constant(0.2)

            optimize = tf.train.AdamOptimizer(
                learning_rate=1e-4)

        return loss, optimize

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.sess.run(self.output, feed_dict={self.input: state})

    # def train(self, critic_input, states, critic_states_input):
    #     loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={self.critic_input: critic_input, critic_states_input: states})
    #     return loss

class TFCritic:
    """Actor (Policy) Model."""

    def __init__(self, session, state_size, action_size, name, actor_input_tensor, checkpoint_file=None):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        self.sess = session
        self.name = name
        self.actor_input_tensor = actor_input_tensor

        if checkpoint_file is None:
            with tf.variable_scope("critic_placeholders_"+self.name):
                self.input = tf.placeholder(tf.float32, shape=(None, state_size), name='input')
                self.actor_input = tf.placeholder(tf.float32, shape=(None, 4), name='actor_input')
                self.y_input = tf.placeholder(tf.float32, shape=(None, 1), name='y_input')


            self.output = self._inference()
            if self.actor_input_tensor is not None:
                self.training_output = self._training_inference()

            self.loss, self.optimizer = self._training_graph()
        else:
            checkpoint_dir = '/'.join(checkpoint_file.split('/')[:-1])
            saver = tf.train.import_meta_graph(checkpoint_file+'.meta')
            saver.restore(self.sess, tf.train.latest_checkpoint(checkpoint_dir))

            self.input = tf.get_default_graph().get_tensor_by_name('critic_placeholders_'+self.name+'/input:0')
            self.actor_input = tf.get_default_graph().get_tensor_by_name('critic_placeholders_'+self.name+'/actor_input:0')
            self.y_input = tf.get_default_graph().get_tensor_by_name('critic_placeholders_'+self.name+'/y_input:0')

            self.loss = tf.get_default_graph().get_tensor_by_name(f'critic_training_{self.name}/loss:0')
            self.optimizer = tf.get_default_graph().get_operation_by_name(f'critic_training_{self.name}/optimize')
            self.output = tf.get_default_graph().get_tensor_by_name(f'critic_inference_{self.name}/dense_2/BiasAdd:0')

        self.step = 0

    def _inference(self):
        with tf.variable_scope("critic_inference_"+self.name):
            layer = tf.layers.batch_normalization(self.input)
            layer = tf.layers.dense(layer, 300, activation=tf.nn.relu)
            layer = tf.layers.batch_normalization(layer)

            concatenated = tf.concat(values=(layer, self.actor_input), axis=1)

            layer = tf.layers.dense(concatenated, 200, activation=tf.nn.relu)
            layer = tf.layers.batch_normalization(layer)
            output = tf.layers.dense(layer, 1)
        return output

    def _training_inference(self):
        with tf.variable_scope("critic_inference_"+self.name, reuse=True):
            layer = tf.layers.batch_normalization(self.input)
            layer = tf.layers.dense(layer, 300, activation=tf.nn.relu)
            layer = tf.layers.batch_normalization(layer)

            concatenated = tf.concat(values=(layer, self.actor_input_tensor), axis=1)

            layer = tf.layers.dense(concatenated, 200, activation=tf.nn.relu)
            layer = tf.layers.batch_normalization(layer)
            output = tf.layers.dense(layer, 1)
        return output

    def _training_graph(self):
        with tf.variable_scope('critic_training_'+self.name):
            loss = tf.losses.mean_squared_error(
                labels=self.y_input, predictions=self.output)
            loss = tf.reduce_mean(loss, name='loss')

            optimize = tf.train.AdamOptimizer(
                learning_rate=1e-3).minimize(loss, name='optimize')

        return loss, optimize

    def forward(self, state, actions):
        """Build a network that maps state -> action values."""
        return self.sess.run(self.output, feed_dict={self.input: state, self.actor_input: actions})
    
    def train(self, states, actions, y_correct):
        loss, _ = self.sess.run([self.loss, self.optimizer], feed_dict={
            self.input: states, self.y_input: y_correct, self.actor_input: actions})
        return loss