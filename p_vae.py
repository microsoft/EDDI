import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
from tensorflow.contrib.distributions import Normal
import re
import copy


class PN_Plus_VAE(object):
    def __init__(self,
                 encoder,
                 decoder,
                 obs_dim,
                 decoder_path,
                 encoder_path,
                 learning_rate=1e-3,
                 optimizer=tf.train.AdamOptimizer,
                 obs_distrib="Gaussian",
                 obs_std=0.1*np.sqrt(2),
                 K = 20,
                 latent_dim = 10,
                 batch_size = 100,
                 load_model=0,
                 M=5,
                 all=1):
        '''
        :param encoder: type of encoder model choosen from coding.py
        :param decoder: type of decoder model choosen from coding.py
        :param obs_dim: maximum number of partial observational dimensions
        :param encoder_path: path for saving encoder model parameter
        :param decoder_path: path for saving decoder model parameter
        :param learning_rate: optimizer learning rate
        :param optimizer: we use Adam here.
        :param obs_distrib: Bernoulli or Gaussian.
        :param obs_std: observational noise for decoder.
        :param K: length of code for summarizing partial observations
        :param latent_dim: latent dimension of VAE
        :param batch_size: training batch size
        :param load_model: 1 = load a pre-trained model from decoder_path and encoder_path
        :param M : number of MC samples used when performing imputing/prediction


        '''
        self._K = K
        self._latent_dim = latent_dim
        self._batch_size = batch_size
        self._encode = encoder
        self._decode = decoder
        self._obs_dim = obs_dim
        self._learning_rate = learning_rate
        self._optimizer = optimizer
        self._obs_distrib = obs_distrib
        self._obs_std = obs_std
        self._load_model = load_model
        self._all = all
        self._decoder_path = decoder_path
        self._encoder_path = encoder_path
        self._M = M
        self._build_graph()


    ## build partial VAE
    def _build_graph(self):

        with tf.variable_scope('is'):
            # placeholder for UCI inputs
            self.x = tf.placeholder(tf.float32, shape=[None, self._obs_dim])
            self.x_flat = tf.reshape(self.x, [-1, 1])
            # placeholder for masks
            self.mask = tf.placeholder(tf.float32, shape=[None, self._obs_dim])
            self._batch_size = tf.shape(self.x)[0]

            # encode inputs (map to parameterization of diagonal Gaussian)
            with tf.variable_scope('encoder'):
                # the tensor F stores ID matrix
                self.F = tf.get_variable(
                    "F",
                    shape=[1, self._obs_dim, 10],
                    initializer=tf.contrib.layers.xavier_initializer())
                self.F = tf.tile(self.F, [self._batch_size, 1, 1])
                self.F = tf.reshape(self.F, [-1, 10])

                self.b = tf.get_variable(
                    "b",
                    shape=[1, self._obs_dim, 1],
                    initializer=tf.contrib.layers.xavier_initializer())
                # bias vector
                self.b = tf.tile(self.b, [self._batch_size, 1, 1])
                self.b = tf.reshape(self.b, [-1, 1])

                self.x_aug = tf.concat(
                    [self.x_flat, self.x_flat * self.F, self.b], 1)
                self.encoded = layers.fully_connected(self.x_aug, self._K)

                self.encoded = tf.reshape(self.encoded,
                                          [-1, self._obs_dim, self._K])
                self.mask_on_hidden = tf.reshape(self.mask,
                                                 [-1, self._obs_dim, 1])
                self.mask_on_hidden = tf.tile(self.mask_on_hidden,
                                              [1, 1, self._K])
                self.encoded = tf.nn.relu(
                    tf.reduce_sum(self.encoded * self.mask_on_hidden, 1))
                self.encoded = layers.fully_connected(self.encoded, 500)
                self.encoded = layers.fully_connected(self.encoded, 200)
                self.encoded = layers.fully_connected(
                    self.encoded, 2 * self._latent_dim, activation_fn=None)

            with tf.variable_scope('sampling'):
                # unpacking mean and (diagonal) variance of latent variable

                self.mean = self.encoded[:, :self._latent_dim]
                self.logvar = self.encoded[:, self._latent_dim:]
                # also calculate standard deviation for practical use
                self.stddev = tf.sqrt(tf.exp(self.logvar))

                # sample from latent space
                epsilon = tf.random_normal(
                    [self._batch_size, self._latent_dim])
                self.z = self.mean + self.stddev * epsilon

            # decode batch
            with tf.variable_scope('generator'):
                self.decoded, _ = self._decode(self.z, self._obs_dim)
            with tf.variable_scope('loss'):
                # KL divergence between approximate posterior q and prior p
                with tf.variable_scope('kl-divergence'):
                    self.kl = self._kl_diagnormal_stdnormal(
                        self.mean, self.logvar)
                # loss likelihood
                if self._obs_distrib == 'Bernoulli':
                    with tf.variable_scope('bernoulli'):
                        self.log_like = self._bernoulli_log_likelihood(
                            self.x, self.decoded, self.mask)
                else:
                    with tf.variable_scope('gaussian'):
                        self.log_like = self._gaussian_log_likelihood(
                            self.x * self.mask, self.decoded * self.mask,
                            self._obs_std)

                self._loss = (self.kl + self.log_like) / tf.cast(
                    self._batch_size, tf.float32)  # loss per instance (actual loss used)
                self._loss_print = (self.kl + self.log_like) / tf.reduce_sum(
                    self.mask)  # loss per feature, for tracking training process only

            with tf.variable_scope('optimizer'):
                optimizer = self._optimizer(learning_rate=self._learning_rate)

            with tf.variable_scope('training-step'):
                self._train = optimizer.minimize(self._loss)



            if self._load_model == 1:
                generator_variables = []
                for v in tf.trainable_variables():
                    if "generator" in v.name:
                        generator_variables.append(v)

                encoder_variables = []
                for v in tf.trainable_variables():
                    if "encoder" in v.name:
                        encoder_variables.append(v)

                # start tensorflow session
                self._sesh = tf.Session()


                load_encoder = tf.contrib.framework.assign_from_checkpoint_fn(
                    self._encoder_path, encoder_variables)
                load_encoder(self._sesh)
                load_generator = tf.contrib.framework.assign_from_checkpoint_fn(
                    self._decoder_path, generator_variables)
                load_generator(self._sesh)

                uninitialized_vars = []
                for var in tf.all_variables():
                    try:
                        self._sesh.run(var)
                    except tf.errors.FailedPreconditionError:
                        uninitialized_vars.append(var)

                init_new_vars_op = tf.variables_initializer(uninitialized_vars)
                self._sesh.run(init_new_vars_op)
            else:
                self._sesh = tf.Session()
                init = tf.global_variables_initializer()
                self._sesh.run(init)

    ## KL divergence
    def _kl_diagnormal_stdnormal(self, mu, log_var):
        '''
        This function calculates KL divergence
        :param mu: mean
        :param log_var: log variance
        :return:
        '''

        var = tf.exp(log_var)
        kl = 0.5 * tf.reduce_sum(tf.square(mu) + var - 1. - log_var)
        return kl


    ## likelihood terms
    @staticmethod
    def _bernoulli_log_likelihood(targets, outputs, mask, eps=1e-8):
        '''
        This function comptutes negative log likelihood for Bernoulli likelihood
        :param targets: test data
        :param outputs: model predictions
        :param mask: mask of missingness
        :return: negative log llh
        '''
        eps = 0
        log_like = -tf.reduce_sum(targets * (tf.log(outputs + eps) * mask) +
                                  (1. - targets) *
                                  (tf.log((1. - outputs) + eps) * mask))

        return log_like

    @staticmethod
    def _gaussian_log_likelihood(targets, mean, std):
        '''
        This function computes negative log likelihood for Gaussians during training
        Note that constant terms are dropped.
        :param targets: test data
        :param mean: mean
        :param std: sigma
        :return: negative log llh
        '''
        se = tf.reduce_sum(
            0.5 * tf.square(targets - mean) / tf.cast(tf.square(std), tf.float32) + tf.cast(tf.log(std), tf.float32))
        return se

    ## optimization function
    def update(self, x, mask):
        '''
        This function is used to update the model during training/optimization
        :param x: training data
        :param mask: mask that indicates observed data and missing data locations
        '''

        _, loss = self._sesh.run([self._train, self._loss_print],
                                 feed_dict={
                                     self.x: x,
                                     self.mask: mask
                                 })
        return loss

    ##
    def full_batch_loss(self, x,mask):
        '''
        retrieve different components of loss function
        :param x: dat matrix
        :param mask: mask that indicates observed data and missing data locations
        :return: overall loss (averaged over all entries), KL term, and reconstruction loss
        '''

        # mask = np.ones((x.shape[0], self._obs_dim))

        loss, kl, recon = self._sesh.run(
            [self._loss_print, self.kl, self.log_like],
            feed_dict={
                self.x: x,
                self.mask: mask
            })
        return loss, kl, recon

    ## predictive likelihood and uncertainties
    def predictive_loss(self, x, mask, eval,M):
        '''
        This function computes predictive losses (negative llh).
        This is used for active learning phase.
        We assume that the last column of x is the target variable of interest
        :param x: data matrix, the last column of x is the target variable of interest
        :param mask: mask that indicates observed data and missing data locations
        :param eval: evaluation metric of active learning. 'rmse':rmse; 'nllh':negative log likelihood
        :return: MAE and RMSE
        '''
        if eval == 'rmse':
            mse = 0
            uncertainty_data = np.zeros((x.shape[0], M))
            for m in range(M):
                decoded = self._sesh.run(
                    self.decoded, feed_dict={
                        self.x: x,
                        self.mask: mask
                    })

                target = x[:, -1]
                output = decoded[:, -1]
                uncertainty_data[:, m] = decoded[:, -1]
                mse += np.square(target - output)

            uncertainty = uncertainty_data.std(axis=1)

            loss = mse / M
        else:
            llh = 0
            lh = 0
            uncertainty_data = np.zeros((x.shape[0], M))
            for m in range(M):
                decoded = self._sesh.run(
                    self.decoded, feed_dict={
                        self.x: x,
                        self.mask: mask
                    })

                target = x[:, -1]
                output = decoded[:, -1]
                uncertainty_data[:, m] = decoded[:, -1]
                if self._obs_distrib == 'Bernoulli':
                    llh += target * (np.log(output + 1e-8)) + (1. - target) * (
                        np.log((1. - output) + 1e-8))
                else:
                    lh += np.exp(-0.5 * np.square(target - output) / (
                        np.square(self._obs_std)) - np.log(self._obs_std) - 0.5 * np.log(2 * np.pi))

            uncertainty = uncertainty_data.std(axis=1)

            if self._obs_distrib == 'Bernoulli':
                nllh = -llh / M
            else:
                nllh = -np.log(lh / M)

            loss = nllh


        return loss, uncertainty

    def impute_losses(self, x, mask_obs, mask_target):
        '''
        This function computes imputation losses
        :param x: data matrix
        :param mask_obs: mask that indicates observed data and missing data locations
        :param mask_target: mask that indicates the test data locations
        :return: squared error (SE) and RMSE
        '''

        SE = 0
        RMSE = 0
        for m in range(self._M):
            decoded = self._sesh.run(self.decoded,
                                     feed_dict={self.x: x, self.mask: mask_obs})

            target = x * mask_target
            output = decoded * mask_target
            SE += np.sum(np.square(target - output))
            RMSE += np.sqrt(np.sum(np.square(target - output)) / np.sum(mask_target))

        SE = SE / self._M
        RMSE = RMSE / self._M
        return SE, RMSE
    def get_imputation(self, x, mask_obs):
        '''
        This function returns the mean of imputation samples from partial vae
        :param x: data matrix
        :param mask_obs: mask that indicates observed data and missing data locations
        :return: mean of imputation samples from partial vae
        '''
        decs = []
        for m in range(self._M):
           decoded = self._sesh.run(self.decoded,
           feed_dict={self.x: x, self.mask: mask_obs})
           decs.append(decoded)
        return np.stack(decs).mean(axis=0)

    ## generate partial inference samples
    def im(self, x, mask):
        '''
        This function produces simulations of unobserved variables based on observed ones.
        :param x: data matrix
        :param mask: mask that indicates observed data and missing data locations
        :return: im, which contains samples of completion.
        '''

        m, v = self._sesh.run([self.mean, self.stddev],
                              feed_dict={
                                  self.x: x,
                                  self.mask: mask
                              })

        ep = np.random.normal(0, 1, [x.shape[0], self._latent_dim])
        z = m + v * ep
        im = self._sesh.run(self.decoded, feed_dict={self.z: z})

        return im

    ## calculate the first term of information reward approximation
    def chaini_I(self, x, mask, i):
        '''
        calculate the first term of information reward approximation
        used only in active learning phase
        :param x: data
        :param mask: mask of missingness
        :param i: indicates the index of x_i
        :return:  the first term of information reward approximation
        '''
        temp_mask = copy.deepcopy(mask)
        m, v = self._sesh.run([self.mean, self.stddev],
                              feed_dict={
                                  self.x: x,
                                  self.mask: temp_mask
                              })

        var = v**2
        log_var = 2 * np.log(v)

        temp_mask[:, i] = 1

        m_i, v_i = self._sesh.run([self.mean, self.stddev],
                                  feed_dict={
                                      self.x: x,
                                      self.mask: temp_mask
                                  })

        var_i = v_i**2
        log_var_i = 2 * np.log(v_i)
        kl_i = 0.5 * np.sum(
            np.square(m_i - m) / v + var_i / var - 1. - log_var_i + log_var,
            axis=1)

        return kl_i

    ## calculate the second term of information reward approximation
    def chaini_II(self, x, mask, i):
        '''
        calculate the second term of information reward approximation
        used only in active learning phase
        Note that we assume that the last column of x is the target variable of interest
        :param x: data
        :param mask: mask of missingness
        :param i: indicates the index of x_i
        :return:  the second term of information reward approximation
        '''
        # mask: represents x_o
        # targets: 0 by M vector, contains M samples from p(\phi|x_o)
        # x : 1 by obs_dim vector, contains 1 instance of data
        # i: indicates the index of x_i
        temp_mask = copy.deepcopy(mask)
        temp_mask[:, -1] = 1
        m, v = self._sesh.run([self.mean, self.stddev],
                              feed_dict={
                                  self.x: x,
                                  self.mask: temp_mask
                              })

        var = v**2
        log_var = 2 * np.log(v)

        temp_mask[:, i] = 1

        m_i, v_i = self._sesh.run([self.mean, self.stddev],
                                  feed_dict={
                                      self.x: x,
                                      self.mask: temp_mask
                                  })

        var_i = v_i**2
        log_var_i = 2 * np.log(v_i)
        kl_i = 0.5 * np.sum(
            np.square(m_i - m) / v + var_i / var - 1. - log_var_i + log_var,
            axis=1)

        return kl_i


    ## save model
    def save_generator(self, path, prefix="is/generator"):
        '''
        This function saves generator parameters to path
        '''
        variables = tf.trainable_variables()
        var_dict = {}
        for v in variables:
            if "generator" in v.name:
                name = prefix + re.sub("is/generator", "", v.name)
                name = re.sub(":0", "", name)
                var_dict[name] = v
        for k, v in var_dict.items():
            print(k)
            print(v)

        saver = tf.train.Saver(var_dict)
        saver.save(self._sesh, path)

    def save_encoder(self, path, prefix="is/encoder"):
        '''
        This function saves encoder parameters to path
        '''
        variables = tf.trainable_variables()
        var_dict = {}
        for v in variables:
            if "encoder" in v.name:
                name = prefix + re.sub("is/encoder", "", v.name)
                name = re.sub(":0", "", name)
                var_dict[name] = v
        for k, v in var_dict.items():
            print(k)
            print(v)

        saver = tf.train.Saver(var_dict)
        saver.save(self._sesh, path)


### function to generate new samples conditioned on observations
def completion(x, mask, M, vae):
    '''
    function to generate new samples conditioned on observations
    :param x: underlying partially observed data
    :param mask: mask of missingness
    :param M: number of MC samples
    :param vae: a pre-trained vae
    :return: sampled missing data, a M by N by D matrix, where M is the number of samples.
    '''

    im = np.zeros((M, x.shape[0], x.shape[1]))

    for m in range(M):
        #tf.reset_default_graph()
        np.random.seed(42 + m)  ### added for bar plots only
        im[m, :, :] = vae.im(x, mask)

    return im

### function for computing reward function approximation
def R_lindley_chain(i, x, mask, M, vae, im, loc):
    '''
    function for computing reward function approximation
    :param i: indicates the index of x_i
    :param x: data matrix
    :param mask: mask of missingness
    :param M: number of MC samples
    :param vae: a pre-trained vae
    :param im: sampled missing data, a M by N by D matrix, where M is the number of samples.
    :return:
    '''

    im_i = im[:, :, i]
    #print(im_i)
    approx_KL = 0
    im_target = im[:, :, -1]
    temp_x = copy.deepcopy(x)
    for m in range(M):
        temp_x[loc, i] = im_i[m, loc]
        KL_I = vae.chaini_I(temp_x[loc, :], mask[loc, :], i)
        temp_x[loc, -1] = im_target[m, loc]
        KL_II = vae.chaini_II(temp_x[loc, :], mask[loc, :], i)

        approx_KL += KL_I
        approx_KL -= KL_II

    R = approx_KL / M

    return R
