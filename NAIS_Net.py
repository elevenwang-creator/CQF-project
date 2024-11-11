"""
CQF Final Project-Deep Neural Networks for Solving High
Dimensional PDEs in Quantitative Finance (DeepPDE)

LANGTAO WANG
"""

# basic imports
import os, random
import numpy as np
import datetime as dt
import time
from abc import ABC, abstractmethod

# warnings
import warnings

warnings.filterwarnings('ignore')

# tensorflow
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

#tf.debugging.enable_check_numerics()
tf.keras.backend.clear_session()
#tf.debugging.set_log_device_placement(True)
#tf.keras.config.set_floatx('float32')
#tf.keras.mixed_precision.set_global_policy('float32')
#tf.config.run_functions_eagerly(True)
#logdir = './tensorboard/debugger'

#tf.debugging.experimental.enable_dump_debug_info(logdir1, tensor_debug_mode="FULL_HEALTH", circular_buffer_size=-1)
# 启动 Profiler
#tf.profiler.experimental.start(logdir1)

#tf.profiler.experimental.start(logdir1)

# 启动 profiling
#tf.profiler.experimental.trace_on(graph=True, profiler_outdir=logdir1)
#tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=logdir1)

from tensorflow.keras.models import Model, load_model

from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.layers import Dropout, Dense, Flatten, Layer, Add
from tensorflow.keras import ops

# kerastuner
import keras_tuner as kt
from kerastuner import HyperParameters


class Linear(Layer):
    """Define Linear layers"""

    def __init__(self, in_features, out_features):
        super(Linear, self).__init__()

        self.in_dims = in_features
        self.w = self.add_weight(
            shape=(in_features, out_features),
            initializer='glorot_uniform',
            trainable=True
        )  # weights_initializer='glorot normal'
        self.b = self.add_weight(
            shape=(out_features,), initializer='zeros', trainable=True
        )  # bias_initializer='zeros'

    #@tf.function
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b


class StableLinear(Layer):
    """Define Stable Linear layers"""

    def __init__(self, in_features, out_features):
        super(StableLinear, self).__init__()

        self.epsilon = 0.01
        self.w = self.add_weight(
            shape=(in_features, out_features),
            initializer='glorot_uniform',
            trainable=True
        )  # weights_initializer='glorot normal'

        #self.w = tf.cast(self.weight_constraint(self.w), dtype=tf.float32)

        self.b = self.add_weight(
            shape=(out_features,), initializer='zeros', trainable=True
        )  # bias_initializer='zeros'

    #@tf.function
    def call(self, inputs):
        w = self.weight_constraint(self.w)
        return tf.matmul(inputs, w) + self.b

    #@tf.function
    def weight_constraint(self, weight):
        delta = 1 - 2 * self.epsilon
        RtR = tf.matmul(tf.transpose(weight), weight)
        norm = tf.norm(RtR)

        #if norm > delta: RtR = delta ** (1 / 2) * RtR / (norm ** (1 / 2))
        RtR_new = tf.sqrt(delta) * RtR / tf.sqrt(norm)
        RtR_update = tf.minimum(RtR_new, RtR)
        #def f2(): return RtR
        #RtR = tf.cond(tf.greater(norm, delta), f1, f2)
        A = RtR_update + tf.eye(RtR.shape[0]) * self.epsilon
        return -A


class NaisNet(Model):
    """ Building NAIS_Net network using keras """

    def __init__(self, layers):
        super(NaisNet, self).__init__()

        self.layer1 = Linear(layers[0], layers[1])

        self.layer2 = StableLinear(layers[1], layers[2])
        self.layer2_input = Linear(layers[0], layers[2])

        self.layer3 = StableLinear(layers[2], layers[3])
        self.layer3_input = Linear(layers[0], layers[3])

        self.layer4 = StableLinear(layers[3], layers[4])
        self.layer4_input = Linear(layers[0], layers[4])

        self.layer5 = Linear(layers[4], layers[5])

    # Building block for the NAIS-Net
    #@tf.function
    def call(self, x):
        """Process stages in the block"""

        u = x

        output = self.layer1(u)  # layer 1
        output = tf.sin(output)

        X1 = output
        output = self.layer2(output)  # layer 2
        output = Add()([self.layer2_input(u), output])
        output = tf.sin(output)
        output = Add()([X1, output])

        X2 = output
        output = self.layer3(output)  # layer 3
        output = Add()([self.layer3_input(u), output])
        output = tf.sin(output)
        output = Add()([X2, output])

        X3 = output
        output = self.layer4(output)  # layer 4
        output = Add()([self.layer4_input(u), output])
        output = tf.sin(output)
        output = Add()([X3, output])

        output = self.layer5(output)  # layer 5

        return output


class FBSNN(ABC):
    """Forward-Backward SDEs using NAIS-Net """

    def __init__(self, Xi, T, M, N, D, layers):

        self.Xi = Xi  # initial point

        self.T = T  # terminal time
        self.M = M  # number of trajectories
        self.N = N  # number of time snapshots
        self.D = D  # number of dimensions

        # Initialize NN
        self.model = NaisNet(layers)

    #@tf.function
    #def net_u(self, t, X):  # M x 1, M x D
    #    inputs = tf.concat([t, X], 1)
        #inputs = tf.Variable(inputs)
    #        inputs = tf.cast(inputs, dtype=tf.float32)
    #    with tf.GradientTape(persistent=True) as tape:  # automatic differentiation
    #            tape.watch(u)
            #tape.watch(self.model.trainable_weights)
    #        u = self.model(inputs)  # M x 1
    #           tf.print(f"u: {u}, inputs: {inputs}")
    #, unconnected_gradients=tf.UnconnectedGradients.ZERO
    #    Du = tape.gradient(u, X, output_gradients=tf.ones_like(u), unconnected_gradients=tf.UnconnectedGradients.ZERO)[0]
        #Du = tape.gradient(u, X, output_gradients=tf.ones_like(u))[0]
    #    return u, Du

    #@tf.costum_function
    #def Dg_tf(self, X):  # M x D
    #    g = self.g_tf(X)

    #    with tf.GradientTape(persistent=True) as tape:
    #        g = self.g_tf(X)

            #tape.watch(self.model.trainable_weights)
    #        #, unconnected_gradients=tf.UnconnectedGradients.ZERO
    #    return tape.gradient(g, X, output_gradients=tf.ones_like(g), unconnected_gradients=tf.UnconnectedGradients.ZERO)[0] 
        #tape.gradient(g, X, output_gradients=tf.ones_like(g))[0]  # M x D

    @tf.function
    def net_u(self, t, X):  # M x 1, M x D

        u = self.model(tf.concat([t, X], 1))  # M x 1
        Du = tf.gradients(u, X)[0]  # M x D

        return u, Du

    @tf.function
    def Dg_tf(self, X):  # M x D
        return tf.gradients(self.g_tf(X), X)[0]  # M x D

    #@tf.function
    def loss_function(self, t, W, Xi):

        loss = 0
        X_list = []
        Y_list = []

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]

        X0 = tf.tile(Xi, [self.M, 1])  # M x D
        Y0, Z0 = self.net_u(t0, X0)  # M x 1, M x D

        X_list.append(X0)
        Y_list.append(Y0)

        for n in range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]
            X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + tf.squeeze(
                tf.matmul(self.sigma_tf(t0, X0, Y0), tf.expand_dims(W1 - W0, -1)), axis=[-1])
            Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + tf.reduce_sum(
                Z0 * tf.squeeze(tf.matmul(self.sigma_tf(t0, X0, Y0), tf.expand_dims(W1 - W0, -1))), axis=1,
                keepdims=True)
            Y1, Z1 = self.net_u(t1, X1)

            loss += tf.reduce_sum(tf.pow(Y1 - Y1_tilde, 2))

            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
            Z0 = Z1

            X_list.append(X0)
            Y_list.append(Y0)

        loss += tf.reduce_sum(tf.pow(Y1 - self.g_tf(X1), 2))
        loss += tf.reduce_sum(tf.pow(Z1 - self.Dg_tf(X1), 2))

        X = tf.stack(X_list, axis=1)
        Y = tf.stack(Y_list, axis=1)

        return loss, X, Y, Y[0, 0, 0]

    #@tf.function
    def fetch_minibatch(self):
        T = self.T

        M = self.M
        N = self.N
        D = self.D

        Dt = np.zeros((M, N + 1, 1))  # M x (N+1) x 1
        DW = np.zeros((M, N + 1, D))  # M x (N+1) x D

        dt = T / N

        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))

        t = np.cumsum(Dt, axis=1)  # M x (N+1) x 1
        W = np.cumsum(DW, axis=1)  # M x (N+1) x D

        return t, W

    @tf.function
    def train_step(self, optimizers, t, winner, Xi):
        # Optimizers
        optimizer = optimizers

        with tf.GradientTape() as tape:
            loss_value, X_pred, Y_pred, Y0_pred = self.loss_function(t, winner, Xi)

        grads = tape.gradient(loss_value, self.model.trainable_weights)
        optimizer.apply(grads, self.model.trainable_weights)

        return loss_value, Y0_pred

    #@tf.function
    def train(self, N_Iter, learning_rate):

        #loss_temp = np.array([])

        # Optimizers
        self.optimizer = Adam(learning_rate=learning_rate)

        start_time = time.time()
        ini_time = time.time()

        for it in range(N_Iter):
            #tf.summary.trace_on(graph=True, profiler=True, profiler_outdir=logdir1)
            t_batch, W_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x D
            t_batch = tf.convert_to_tensor(t_batch, dtype=tf.float32)
            W_batch = tf.convert_to_tensor(W_batch, dtype=tf.float32)
            self.Xi = tf.convert_to_tensor(self.Xi, dtype=tf.float32)

            loss_value, Y0_pred = self.train_step(self.optimizer, t_batch, W_batch, self.Xi)

            #with train_summary_writer1.as_default():
            #   tf.summary.trace_export(name="my_func_trace", step=it)
            if it % 100 == 0:
                elapsed = time.time() - start_time
                total_time = time.time() - ini_time
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Total Time: %.2f, Learning Rate: %.3e' %
                      (it, loss_value, Y0_pred, elapsed, total_time, learning_rate))
                start_time = time.time()

            logdir1 = os.path.join("./tensorboard/logs", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
            train_summary_writer1 = tf.summary.create_file_writer(logdir1)
            with train_summary_writer1.as_default():
                tf.summary.scalar('loss', loss_value, step=100)

        # Specify callback functions
        #model_path = './model/model.keras'
        #logdir = os.path.join("./tensorboard/logs", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))

        #my_callbacks = [ModelCheckpoint(filepath=model_path, verbose=1, monitor='loss', mode='min', save_best_only=True),TensorBoard(log_dir=logdir, histogram_freq=1)]

        #self.model.fit(self.Xi, epochs=N_Iter, verbose=1, callbacks=my_callbacks)

    def predict(self, Xi_star, t_star, W_star):
        t_star = tf.convert_to_tensor(t_star, dtype=tf.float32)
        W_star = tf.convert_to_tensor(W_star, dtype=tf.float32)
        Xi_star = tf.convert_to_tensor(Xi_star, dtype=tf.float32)

        loss, X_star, Y_star, Y0_pred = self.loss_function(t_star, W_star, Xi_star)

        return X_star, Y_star

    ###########################################################################
    ############################# Change Here! ################################
    ###########################################################################
    @abstractmethod
    def phi_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        pass  # M x1

    @abstractmethod
    def g_tf(self, X):  # M x D
        pass  # M x 1

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):  # M x 1, M x D, M x 1, M x D
        M = self.M
        D = self.D
        return np.zeros([M, D])  # M x D

    @abstractmethod
    def sigma_tf(self, t, X, Y):  # M x 1, M x D, M x 1
        M = self.M
        D = self.D
        return tf.linalg.diag(tf.ones([M, D]))  # M x D x D
    ###########################################################################