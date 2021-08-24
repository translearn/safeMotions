# Code  adapted from https://github.com/ray-project/ray/blob/master/rllib/models/tf/tf_action_dist.py
# Copyright 2021 Ray Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import gym
from math import log
from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.tf.tf_action_dist import TFActionDistribution
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_tf, try_import_tfp
from ray.rllib.utils.typing import TensorType, List, Union, ModelConfigDict
from ray.rllib.utils import SMALL_NUMBER

tf1, tf, tfv = try_import_tf()
tfp = try_import_tfp()


class TruncatedNormal(TFActionDistribution):
    """Normal distribution that is truncated such that all values are within [low, high].
    The distribution is defined by the mean and the std deviation of a normal distribution that is not truncated.
    (loc=mean -> first half of input, scale=exp(log_std) -> log_std = second half of input)
    KL corresponds to the KL of the underlying normal distributions
    """

    def __init__(self, inputs: List[TensorType], model: ModelV2, low: float = -1.0,
                 high: float = 1.0):
        mean_normal, log_std_normal = tf.split(inputs, 2, axis=1)
        self.mean_normal = mean_normal
        self.log_std_normal = log_std_normal
        self.std_normal = tf.exp(log_std_normal)
        self.low = low
        self.high = high
        self.zeros = tf.reduce_sum(tf.zeros_like(self.mean_normal), axis=1)
        self.dist = tfp.distributions.TruncatedNormal(
            loc=self.mean_normal, scale=self.std_normal,
            low=self.low, high=self.high)
        super().__init__(inputs, model)

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        return self.dist.mean()

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        return tf.math.reduce_sum(self.dist.log_prob(x), axis=-1)

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution) -> TensorType:
        assert isinstance(other, TruncatedNormal)
        # return self.dist.cross_entropy(other.dist) - self.dist.entropy()  -> kl_divergence not implemented
        # return tf.reduce_sum(self.dist.kl_divergence(other.dist), axis=1)  -> kl_divergence not implemented
        # Return the kl_divergence of the underlying normal distributions since the correct kl_divergence is not
        # implemented
        return tf.reduce_sum(
            other.log_std_normal - self.log_std_normal +
            (tf.math.square(self.std_normal) + tf.math.square(self.mean_normal - other.mean_normal))
            / (2.0 * tf.math.square(other.std_normal)) - 0.5,
            axis=1)

    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        return tf.reduce_sum(self.dist.entropy(), axis=1)

    @override(TFActionDistribution)
    def _build_sample_op(self) -> TensorType:
        return self.dist.sample()

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
            action_space: gym.Space,
            model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape) * 2


class TruncatedNormalZeroKL(TruncatedNormal):
    """Normal distribution that is truncated such that all values are within [low, high].
    The distribution is defined by the mean and the std deviation of a normal distribution that is not truncated.
    (loc=mean -> first half of input, scale=exp(log_std) -> log_std = second half of input).
    KL is always set to zero.
    """

    def __init__(self, inputs: List[TensorType], model: ModelV2, low: float = -1.0,
                 high: float = 1.0):
        super().__init__(inputs, model, low, high)
        self.zeros = tf.reduce_sum(tf.zeros_like(self.mean_normal), axis=1)

    @override(TruncatedNormal)
    def kl(self, other: ActionDistribution) -> TensorType:
        assert isinstance(other, TruncatedNormal)
        return self.zeros


class BetaBase(TFActionDistribution):
    """
    A Beta distribution is defined on the interval [0, 1] and parameterized by
    shape parameters alpha and beta (also called concentration parameters).
    PDF(x; alpha, beta) = x**(alpha - 1) (1 - x)**(beta - 1) / Z
        with Z = Gamma(alpha) Gamma(beta) / Gamma(alpha + beta)
        and Gamma(n) = (n - 1)!
    """

    def __init__(self,
                 inputs: List[TensorType],
                 model: ModelV2,
                 low: float = -1.0,
                 high: float = 1.0):

        self.dist = None
        self.low = low
        self.high = high

    @override(ActionDistribution)
    def deterministic_sample(self) -> TensorType:
        mean = self.dist.mean()
        return self._squash(mean)

    @override(TFActionDistribution)
    def _build_sample_op(self) -> TensorType:
        return self._squash(self.dist.sample())

    @override(ActionDistribution)
    def entropy(self) -> TensorType:
        return tf.reduce_sum(self.dist.entropy(), axis=1)

    @override(ActionDistribution)
    def kl(self, other: ActionDistribution) -> TensorType:
        assert isinstance(other, BetaBase)
        return tf.reduce_sum(self.dist.kl_divergence(other.dist), axis=1)

    @override(ActionDistribution)
    def logp(self, x: TensorType) -> TensorType:
        unsquashed_values = self._unsquash(x)
        return tf.math.reduce_sum(
            self.dist.log_prob(unsquashed_values), axis=-1)

    def _squash(self, raw_values: TensorType) -> TensorType:
        return raw_values * (self.high - self.low) + self.low

    def _unsquash(self, values: TensorType) -> TensorType:
        return (values - self.low) / (self.high - self.low)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(
            action_space: gym.Space,
            model_config: ModelConfigDict) -> Union[int, np.ndarray]:
        return np.prod(action_space.shape) * 2


class BetaMeanTotal(BetaBase):
    """
    A Beta distribution defined by the mean (squashed) and the log total concentration
    """

    def __init__(self,
                 inputs: List[TensorType],
                 model: ModelV2,
                 low: float = -1.0,
                 high: float = 1.0):
        super().__init__(inputs, model, low, high)
        mean_squashed, log_total_concentration = tf.split(self.inputs, 2, axis=-1)
        log_total_concentration = tf.clip_by_value(log_total_concentration, log(SMALL_NUMBER),
                                                   -log(SMALL_NUMBER))
        # total_concentration > 0
        mean = self._unsquash(mean_squashed)
        total_concentration = tf.exp(log_total_concentration)
        alpha = mean * total_concentration
        beta = (1.0 - mean) * total_concentration
        self.dist = tfp.distributions.Beta(
            concentration1=alpha, concentration0=beta)
        super(BetaBase, self).__init__(inputs, model)


class BetaAlphaBeta(BetaBase):
    """
    A Beta distribution defined by alpha and beta with alpha, beta > 1
    """

    def __init__(self,
                 inputs: List[TensorType],
                 model: ModelV2,
                 low: float = -1.0,
                 high: float = 1.0):
        inputs = tf.clip_by_value(inputs, log(SMALL_NUMBER),
                                  -log(SMALL_NUMBER))
        inputs = tf.math.log(tf.math.exp(inputs) + 1.0) + 1.0  # ensures alpha > 1, beta > 1
        super().__init__(inputs, model, low, high)
        alpha, beta = tf.split(inputs, 2, axis=-1)

        self.dist = tfp.distributions.Beta(
            concentration1=alpha, concentration0=beta)
        super(BetaBase, self).__init__(inputs, model)

