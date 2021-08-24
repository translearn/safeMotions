# Code  adapted from https://github.com/ray-project/ray/blob/master/rllib/models/tf/fcnet.py
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
from typing import Dict, Optional, Sequence

from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.framework import try_import_tf
from ray.rllib.utils.typing import TensorType, List

tf1, tf, tfv = try_import_tf()


class FullyConnectedNetworkLastLayerActivation(tf.keras.Model if tf else object):
    """Generic fully connected network implemented in tf Keras."""

    def __init__(
            self,
            input_space: gym.spaces.Space,
            action_space: gym.spaces.Space,
            num_outputs: Optional[int] = None,
            *,
            name: str = "",
            fcnet_hiddens: Optional[Sequence[int]] = (),
            fcnet_activation: Optional[str] = None,
            post_fcnet_hiddens: Optional[Sequence[int]] = (),
            post_fcnet_activation: Optional[str] = None,
            no_final_linear: bool = False,
            vf_share_layers: bool = False,
            free_log_std: bool = False,
            last_layer_activation: str = None,
            no_log_std_activation: bool = False,
            output_intermediate_layers: bool = False,
            **kwargs,
    ):
        super().__init__(name=name)

        hiddens = list(fcnet_hiddens or ()) + \
            list(post_fcnet_hiddens or ())
        activation = fcnet_activation
        if not fcnet_hiddens:
            activation = post_fcnet_activation
        activation = get_activation_fn(activation)

        if last_layer_activation is not None:
            last_layer_activation = get_activation_fn(last_layer_activation)

        # Generate free-floating bias variables for the second half of
        # the outputs.
        if free_log_std:
            assert num_outputs % 2 == 0, (
                "num_outputs must be divisible by two", num_outputs)
            num_outputs = num_outputs // 2
            self.log_std_var = tf.Variable(
                [0.0] * num_outputs, dtype=tf.float32, name="log_std")

        # We are using obs_flat, so take the flattened shape as input.
        inputs = tf.keras.layers.Input(
            shape=(int(np.product(input_space.shape)), ), name="observations")
        # Last hidden layer output (before logits outputs).
        last_layer = inputs
        # The action distribution outputs.
        logits_out = None
        logits_intermediate_layers = []
        self._output_intermediate_layers = output_intermediate_layers
        self._intermediate_layer_names = []
        i = 1

        # Create layers 0 to second-last.
        for size in hiddens[:-1]:
            name = "fc_{}".format(i)
            last_layer = tf.keras.layers.Dense(
                size,
                name=name,
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
            self._intermediate_layer_names.append(name)
            logits_intermediate_layers.append(last_layer)
            i += 1

        # The last layer is adjusted to be of size num_outputs, but it's a
        # layer with activation.
        if no_final_linear and num_outputs:
            logits_out = tf.keras.layers.Dense(
                num_outputs,
                name="fc_out",
                activation=activation,
                kernel_initializer=normc_initializer(1.0))(last_layer)
        # Finish the layers with the provided sizes (`hiddens`), plus -
        # iff num_outputs > 0 - a last linear layer of size num_outputs.
        else:
            if len(hiddens) > 0:
                name = "fc_{}".format(i)
                last_layer = tf.keras.layers.Dense(
                    hiddens[-1],
                    name=name,
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0))(last_layer)
                logits_intermediate_layers.append(last_layer)
                self._intermediate_layer_names.append(name)
            if num_outputs:
                if no_log_std_activation and not vf_share_layers:
                    actions_out = tf.keras.layers.Dense(
                        num_outputs // 2,
                        name="fc_out_actions",
                        activation=last_layer_activation,
                        kernel_initializer=normc_initializer(0.01))(last_layer)
                    log_std_out = tf.keras.layers.Dense(
                        num_outputs // 2,
                        name="fc_out_log_std",
                        activation=None,
                        kernel_initializer=normc_initializer(0.01))(last_layer)
                    logits_out = tf.keras.layers.Concatenate(axis=1)([actions_out, log_std_out])
                else:
                    logits_out = tf.keras.layers.Dense(
                        num_outputs,
                        name="fc_out",
                        activation=last_layer_activation,
                        kernel_initializer=normc_initializer(0.01))(last_layer)

        # Concat the log std vars to the end of the state-dependent means.
        if free_log_std and logits_out is not None:

            def tiled_log_std(x):
                return tf.tile(
                    tf.expand_dims(self.log_std_var, 0), [tf.shape(x)[0], 1])

            log_std_out = tf.keras.layers.Lambda(tiled_log_std)(inputs)
            logits_out = tf.keras.layers.Concatenate(axis=1)(
                [logits_out, log_std_out])

        last_vf_layer = None
        vf_intermediate_layers = []
        if not vf_share_layers:
            # Build a parallel set of hidden layers for the value net.
            last_vf_layer = inputs
            i = 1
            for size in hiddens:
                name = "fc_value_{}".format(i)
                last_vf_layer = tf.keras.layers.Dense(
                    size,
                    name=name,
                    activation=activation,
                    kernel_initializer=normc_initializer(1.0))(last_vf_layer)
                i += 1
                vf_intermediate_layers.append(last_vf_layer)
                self._intermediate_layer_names.append(name)

        value_out = tf.keras.layers.Dense(
            1,
            name="value_out",
            activation=None,
            kernel_initializer=normc_initializer(0.01))(
                last_vf_layer if last_vf_layer is not None else last_layer)

        if not self._output_intermediate_layers:
            self.base_model = tf.keras.Model(
                inputs, [(logits_out
                          if logits_out is not None else last_layer), value_out])
        else:
            self.base_model = tf.keras.Model(
                inputs, [logits_layer for logits_layer in logits_intermediate_layers] +
                        [vf_layer for vf_layer in vf_intermediate_layers] +
                        [(logits_out if logits_out is not None else last_layer), value_out])

    def call(self, input_dict: SampleBatch) -> \
            (TensorType, List[TensorType], Dict[str, TensorType]):
        model_out = self.base_model(input_dict[SampleBatch.OBS])
        logits_out = model_out[-2]
        value_out = model_out[-1]
        extra_outs = {SampleBatch.VF_PREDS: tf.reshape(value_out, [-1])}
        if self._output_intermediate_layers:
            for i in range(len(model_out) - 2):
                extra_outs[self._intermediate_layer_names[i]] = model_out[i]
            extra_outs['logits'] = logits_out
        return logits_out, [], extra_outs
