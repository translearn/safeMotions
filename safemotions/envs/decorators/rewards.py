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
    ADAPTATION_MAX_PUNISHMENT = 1.0
    END_MIN_DISTANCE_MAX_PUNISHMENT = 1.0
    END_MAX_TORQUE_MAX_PUNISHMENT = 1.0
    END_MAX_TORQUE_MIN_THRESHOLD = 0.9

    # braking trajectory max punishment (either collision or torque -> max)
    BRAKING_TRAJECTORY_MAX_PUNISHMENT = 1.0
    # braking trajectory torque penalty
    BRAKING_TRAJECTORY_MAX_TORQUE_MIN_THRESHOLD = 0.9  # rel. abs. torque threshold

    def __init__(self,
                 *vargs,
                 normalize_reward_to_frequency=False,
                 normalize_reward_to_initial_target_point_distance=False,
                 punish_action=False,
                 action_punishment_min_threshold=ACTION_THRESHOLD,
                 action_max_punishment=ACTION_MAX_PUNISHMENT,
                 punish_adaptation=False,
                 adaptation_max_punishment=ADAPTATION_MAX_PUNISHMENT,
                 punish_end_min_distance=False,
                 end_min_distance_max_punishment=END_MIN_DISTANCE_MAX_PUNISHMENT,
                 end_min_distance_max_threshold=None,
                 punish_end_max_torque=False,
                 end_max_torque_max_punishment=END_MAX_TORQUE_MAX_PUNISHMENT,
                 end_max_torque_min_threshold=END_MAX_TORQUE_MIN_THRESHOLD,
                 braking_trajectory_max_punishment=BRAKING_TRAJECTORY_MAX_PUNISHMENT,
                 punish_braking_trajectory_min_distance=False,
                 braking_trajectory_min_distance_max_threshold=None,
                 punish_braking_trajectory_max_torque=False,
                 braking_trajectory_max_torque_min_threshold=BRAKING_TRAJECTORY_MAX_TORQUE_MIN_THRESHOLD,
                 target_point_reward_factor=1.0,
                 **kwargs):

        super().__init__(*vargs, **kwargs)

        # reward settings
        self.reward_range = [0, 1]
        self._normalize_reward_to_frequency = normalize_reward_to_frequency
        self._normalize_reward_to_initial_target_point_distance = normalize_reward_to_initial_target_point_distance

        self._punish_action = punish_action
        self._action_punishment_min_threshold = action_punishment_min_threshold
        self._action_max_punishment = action_max_punishment

        self._punish_adaptation = punish_adaptation
        self._adaptation_max_punishment = adaptation_max_punishment

        self._punish_end_min_distance = punish_end_min_distance
        self._end_min_distance_max_punishment = end_min_distance_max_punishment
        self._end_min_distance_max_threshold = end_min_distance_max_threshold
        self._punish_end_max_torque = punish_end_max_torque
        self._end_max_torque_max_punishment = end_max_torque_max_punishment
        self._end_max_torque_min_threshold = end_max_torque_min_threshold

        self._punish_braking_trajectory_min_distance = punish_braking_trajectory_min_distance
        self._braking_trajectory_min_distance_max_threshold = braking_trajectory_min_distance_max_threshold
        self._punish_braking_trajectory_max_torque = punish_braking_trajectory_max_torque
        self._braking_trajectory_max_punishment = braking_trajectory_max_punishment
        self._max_torque_min_threshold = braking_trajectory_max_torque_min_threshold

        self._target_point_reward_factor = target_point_reward_factor

        if self._punish_braking_trajectory_min_distance or self._punish_end_min_distance:
            if self._punish_braking_trajectory_min_distance and \
                    self._braking_trajectory_min_distance_max_threshold is None:
                raise ValueError("punish_braking_trajectory_min_distance requires "
                                 "braking_trajectory_min_distance_max_threshold to be specified")
            if self._punish_end_min_distance and \
                    self._end_min_distance_max_threshold is None:
                raise ValueError("punish_end_min_distance requires "
                                 "end_min_distance_max_threshold to be specified")

            if self._punish_braking_trajectory_min_distance and self._punish_end_min_distance:
                end_min_distance_max_threshold = max(self._braking_trajectory_min_distance_max_threshold,
                                                     self._end_min_distance_max_threshold)
            elif self._punish_braking_trajectory_min_distance:
                end_min_distance_max_threshold = self._braking_trajectory_min_distance_max_threshold
            else:
                end_min_distance_max_threshold = self._end_min_distance_max_threshold

            self._robot_scene.obstacle_wrapper.set_maximum_relevant_distance(end_min_distance_max_threshold)

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

        reward = 0
        target_point_reward = 0
        action_punishment = 0
        adaptation_punishment = 0
        end_min_distance_punishment = 0
        end_max_torque_punishment = 0

        braking_trajectory_min_distance_punishment = 0
        braking_trajectory_max_torque_punishment = 0

        if self._robot_scene.obstacle_wrapper.use_target_points:
            if self._punish_action:
                action_punishment = self._compute_action_punishment()  # action punishment

            if self._punish_adaptation:
                adaptation_punishment = self._adaptation_punishment

            if self._punish_end_min_distance:
                if self._end_min_distance is None:
                    self._end_min_distance, _, _, _ = self._robot_scene.obstacle_wrapper.get_minimum_distance(
                        manip_joint_indices=self._robot_scene.manip_joint_indices,
                        target_position=self._start_position)

                end_min_distance_punishment = self._compute_quadratic_punishment(
                    a=self._end_min_distance_max_threshold,
                    b=self._end_min_distance,
                    c=self._end_min_distance_max_threshold,
                    d=self._robot_scene.obstacle_wrapper.closest_point_safety_distance)

            if self._punish_end_max_torque:
                if self._end_max_torque is not None:
                    # None if check_braking_trajectory is False and asynchronous movement execution is active
                    # in this case, no penality is computed, but the penalty is not required anyways
                    end_max_torque_punishment = self._compute_quadratic_punishment(
                        a=self._end_max_torque,
                        b=self._end_max_torque_min_threshold,
                        c=1,
                        d=self._end_max_torque_min_threshold)

            target_point_reward = self._robot_scene.obstacle_wrapper.get_target_point_reward(
                normalize_distance_reward_to_initial_target_point_distance=
                self._normalize_reward_to_initial_target_point_distance)

            reward = target_point_reward * self._target_point_reward_factor \
                - action_punishment * self._action_max_punishment \
                - adaptation_punishment * self._adaptation_max_punishment \
                - end_min_distance_punishment * self._end_min_distance_max_punishment \
                - end_max_torque_punishment * self._end_max_torque_max_punishment

            if self._punish_braking_trajectory_min_distance or self._punish_braking_trajectory_max_torque:
                braking_trajectory_min_distance_punishment, braking_trajectory_max_torque_punishment = \
                    self._robot_scene.obstacle_wrapper.get_braking_trajectory_punishment(
                        minimum_distance_max_threshold=self._braking_trajectory_min_distance_max_threshold,
                        maximum_torque_min_threshold=self._max_torque_min_threshold)

                if self._punish_braking_trajectory_min_distance and self._punish_braking_trajectory_max_torque:
                    braking_trajectory_punishment = self._braking_trajectory_max_punishment * \
                                                    max(braking_trajectory_min_distance_punishment,
                                                        braking_trajectory_max_torque_punishment)
                elif self._punish_braking_trajectory_min_distance:
                    braking_trajectory_punishment = self._braking_trajectory_max_punishment * \
                                                    braking_trajectory_min_distance_punishment
                else:
                    braking_trajectory_punishment = self._braking_trajectory_max_punishment * \
                                                    braking_trajectory_max_torque_punishment

                reward = reward - braking_trajectory_punishment

        if self._normalize_reward_to_frequency:
            # Baseline: 10 Hz
            reward = reward * self._trajectory_time_step / 0.1

        info['average'].update(reward=reward,
                               action_punishment=action_punishment,
                               adaptation_punishment=adaptation_punishment,
                               end_min_distance_punishment=end_min_distance_punishment,
                               end_max_torque_punishment=end_max_torque_punishment,
                               target_point_reward=target_point_reward,
                               braking_trajectory_min_distance_punishment=braking_trajectory_min_distance_punishment,
                               braking_trajectory_max_torque_punishment=braking_trajectory_max_torque_punishment)

        info['max'].update(reward=reward,
                           action_punishment=action_punishment,
                           adaptation_punishment=adaptation_punishment,
                           end_min_distance_punishment=end_min_distance_punishment,
                           end_max_torque_punishment=end_max_torque_punishment,
                           target_point_reward=target_point_reward,
                           braking_trajectory_min_distance_punishment=braking_trajectory_min_distance_punishment,
                           braking_trajectory_max_torque_punishment=braking_trajectory_max_torque_punishment
                           )

        info['min'].update(reward=reward,
                           action_punishment=action_punishment,
                           adaptation_punishment=adaptation_punishment,
                           end_min_distance_punishment=end_min_distance_punishment,
                           end_max_torque_punishment=end_max_torque_punishment,
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
            logging.warning("Jerk violation: t = %s Joint: %s Rel jerk %s",
                            (self._episode_length - 1) * self._trajectory_time_step,
                            np.argmax(np.abs(curr_joint_jerk_rel)),
                            max_normalized_jerk)

        info['max']['joint_jerk_violation'] = jerk_violation

        logging.debug("Reward %s: %s", self._episode_length - 1, reward)

        return reward, info

    def _compute_quadratic_punishment(self, a, b, c, d):
        # returns max(min((a - b) / (c - d), 1), 0) ** 2
        punishment = (a - b) / (c - d)
        return max(min(punishment, 1), 0) ** 2

    def _compute_action_punishment(self):
        # The aim of the action punishment is to avoid the action being too close to -1 or 1.
        action_abs = np.abs(self._last_action)
        max_action_abs = max(action_abs)
        return self._compute_quadratic_punishment(max_action_abs, self._action_punishment_min_threshold,
                                                  1, self._action_punishment_min_threshold)

