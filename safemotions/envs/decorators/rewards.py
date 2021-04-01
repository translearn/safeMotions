# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
import numpy as np
import logging
from abc import ABC
from safemotions.envs.safe_motions_base import SafeMotionsBase


def normalize_joint_values(values, joint_limits):
    return list(np.array(values) / np.array(joint_limits))


class TargetPointReachingReward(ABC, SafeMotionsBase):
    # optional action penalty
    ACTION_THRESHOLD = 0.9
    ACTION_MAX_PUNISHMENT = 1.0
    # braking trajectory collision penalty
    MIN_DISTANCE_MAX_THRESHOLD = 0.05  # meter; a lower distance will lead to punishments
    MIN_DISTANCE_MAX_PUNISHMENT = 1.0
    # braking trajectory torque penalty
    MAX_TORQUE_MIN_THRESHOLD = 0.9  # rel. abs. torque threshold
    MAX_TORQUE_MAX_PUNISHMENT = 1.0

    def __init__(self,
                 *vargs,
                 normalize_reward_to_frequency=False,
                 punish_action=False,
                 punish_braking_trajectory_min_distance=False,
                 punish_braking_trajectory_max_torque=False,
                 action_punishment_min_threshold=ACTION_THRESHOLD,
                 action_max_punishment=ACTION_MAX_PUNISHMENT,
                 braking_trajectory_min_distance_max_threshold=MIN_DISTANCE_MAX_THRESHOLD,
                 braking_trajectory_min_distance_max_punishment=MIN_DISTANCE_MAX_PUNISHMENT,
                 braking_trajectory_max_torque_min_threshold=MAX_TORQUE_MIN_THRESHOLD,
                 braking_trajectory_max_torque_max_punishment=MAX_TORQUE_MAX_PUNISHMENT,
                 target_point_reward_factor=1.0,
                 **kwargs):
        super().__init__(*vargs, **kwargs)

        # reward settings
        self.reward_range = [0, 1]
        self._normalize_reward_to_frequency = normalize_reward_to_frequency

        self._punish_action = punish_action
        self._action_punishment_min_threshold = action_punishment_min_threshold
        self._action_max_punishment = action_max_punishment

        self._punish_braking_trajectory_min_distance = punish_braking_trajectory_min_distance
        self._punish_braking_trajectory_max_torque = punish_braking_trajectory_max_torque
        self._min_distance_max_threshold = braking_trajectory_min_distance_max_threshold
        self._min_distance_max_punishment = braking_trajectory_min_distance_max_punishment
        self._max_torque_min_threshold = braking_trajectory_max_torque_min_threshold
        self._max_torque_max_punishment = braking_trajectory_max_torque_max_punishment

        self._target_point_reward_factor = target_point_reward_factor

    def _get_reward(self):
        info = {'average': {'reward': 0,  'action_punishment': 0, 'target_point_reward': 0,
                            'braking_trajectory_min_distance_punishment': 0,
                            'braking_trajectory_max_torque_punishment': 0},
                'max': {'reward': 0, 'action_punishment': 0, 'target_point_reward': 0,
                        'braking_trajectory_min_distance_punishment': 0,
                        'braking_trajectory_max_torque_punishment': 0},
                'min': {'reward': 0, 'action_punishment': 0, 'target_point_reward': 0,
                        'braking_trajectory_min_distance_punishment': 0,
                        'braking_trajectory_max_torque_punishment': 0
                        }}

        # action punishment

        reward = 0
        target_point_reward = 0
        action_punishment = 0
        braking_trajectory_min_distance_punishment = 0
        braking_trajectory_max_torque_punishment = 0

        if self._robot_scene.obstacle_wrapper.use_target_points:
            if self._punish_action:
                action_punishment = self._compute_action_punishment()

            target_point_reward = self._robot_scene.obstacle_wrapper.get_target_point_reward()

            if self._punish_braking_trajectory_min_distance or self._punish_braking_trajectory_max_torque:
                braking_trajectory_min_distance_punishment, braking_trajectory_max_torque_punishment = \
                    self._robot_scene.obstacle_wrapper.get_braking_trajectory_punishment(
                        minimum_distance_max_threshold=self._min_distance_max_threshold,
                        maximum_torque_min_threshold=self._max_torque_min_threshold)

            reward = target_point_reward * self._target_point_reward_factor \
                     - action_punishment * self._action_max_punishment

            if self._punish_braking_trajectory_min_distance:
                reward = reward - braking_trajectory_min_distance_punishment * self._min_distance_max_punishment

            if self._punish_braking_trajectory_max_torque:
                reward = reward - braking_trajectory_max_torque_punishment * self._max_torque_max_punishment

        if self._normalize_reward_to_frequency:
            # Baseline: 10 Hz
            reward = reward * self._trajectory_time_step / 0.1

        info['average'].update(reward=reward,
                               action_punishment=action_punishment,
                               target_point_reward=target_point_reward,
                               braking_trajectory_min_distance_punishment=braking_trajectory_min_distance_punishment,
                               braking_trajectory_max_torque_punishment=braking_trajectory_max_torque_punishment)

        info['max'].update(reward=reward,
                           action_punishment=action_punishment,
                           target_point_reward=target_point_reward,
                           braking_trajectory_min_distance_punishment=braking_trajectory_min_distance_punishment,
                           braking_trajectory_max_torque_punishment=braking_trajectory_max_torque_punishment
                           )

        info['min'].update(reward=reward,
                           action_punishment=action_punishment,
                           target_point_reward=target_point_reward,
                           braking_trajectory_min_distance_punishment=braking_trajectory_min_distance_punishment,
                           braking_trajectory_max_torque_punishment=braking_trajectory_max_torque_punishment
                           )

        # add information about the jerk as custom metric
        curr_joint_jerk = (np.array(self._get_generated_trajectory_point(-1, key='accelerations'))
                           - np.array(self._get_generated_trajectory_point(-2, key='accelerations'))) \
                          / self._trajectory_time_step

        curr_joint_jerk_rel = normalize_joint_values(curr_joint_jerk, self._robot_scene.max_jerk_linear_interpolation)
        jerk_violation = 0.0

        for j in range(self._num_manip_joints):
            info['average']['joint_{}_jerk'.format(j)] = curr_joint_jerk_rel[j]
            info['average']['joint_{}_jerk_abs'.format(j)] = abs(curr_joint_jerk_rel[j])
            info['max']['joint_{}_jerk'.format(j)] = curr_joint_jerk_rel[j]
            info['min']['joint_{}_jerk'.format(j)] = curr_joint_jerk_rel[j]

        max_normalized_jerk = np.max(np.abs(curr_joint_jerk_rel))
        if max_normalized_jerk > 1.002:
            jerk_violation = 1.0
            logging.warning("Jerk limit exceeded: %s", max_normalized_jerk)

        info['max']['joint_jerk_violation'] = jerk_violation

        return reward, info

    def _compute_action_punishment(self):
        # The aim of the action punishment is to avoid the action being too close to -1 or 1.
        action_abs = np.abs(self._last_action)
        max_action_abs = max(action_abs)
        action_punishment = (max_action_abs - self._action_punishment_min_threshold) / \
                            (1 - self._action_punishment_min_threshold)

        return max(min(action_punishment, 1), 0) ** 2

