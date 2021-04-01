# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
from abc import ABC
import numpy as np
from gym.spaces import Box
from safemotions.envs.safe_motions_base import SafeMotionsBase


def normalize_joint_values(values, joint_limits):
    return list(np.array(values) / np.array(joint_limits))


class SafeObservation(ABC, SafeMotionsBase):

    def __init__(self,
                 *vargs,
                 m_prev=1,
                 obs_add_target_point_pos=False,
                 obs_add_target_point_relative_pos=False,
                 **kwargs):
        super().__init__(*vargs, **kwargs)

        self._m_prev = m_prev
        self._next_joint_acceleration_mapping = None

        self._obs_add_target_point_pos = obs_add_target_point_pos
        self._obs_add_target_point_relative_pos = obs_add_target_point_relative_pos

        obs_current_size = 3  # pos, vel, acc

        obs_target_point_size = 0

        if self._robot_scene.obstacle_wrapper.use_target_points:
            if obs_add_target_point_pos:
                obs_target_point_size += 3 * self._robot_scene.num_robots

            if obs_add_target_point_relative_pos:
                obs_target_point_size += 3 * self._robot_scene.num_robots

            if self._robot_scene.obstacle_wrapper.target_point_sequence != 0:
                obs_target_point_size += self._robot_scene.num_robots  # target point active signal for each robot

        self._observation_size = self._m_prev * self._num_manip_joints + obs_current_size * self._num_manip_joints \
                                 + obs_target_point_size

        self.observation_space = Box(low=np.float32(-1), high=np.float32(1), shape=(self._observation_size,),
                                     dtype=np.float32)

        logging.info("Observation size: " + str(self._observation_size))

    def reset(self):
        super().reset()
        self._robot_scene.prepare_for_start_of_episode()
        self._init_observation_attributes()

        observation, observation_info = self._get_observation()

        return observation

    def _init_observation_attributes(self):
        pass

    def _prepare_for_next_action(self):
        super()._prepare_for_next_action()

    def _get_observation(self):
        prev_joint_accelerations = self._get_m_prev_joint_values(self._m_prev, key='accelerations')
        curr_joint_position = self._get_generated_trajectory_point(-1)
        curr_joint_velocity = self._get_generated_trajectory_point(-1, key='velocities')
        curr_joint_acceleration = self._get_generated_trajectory_point(-1, key='accelerations')
        prev_joint_accelerations_rel = [normalize_joint_values(p, self._robot_scene.max_accelerations)
                                        for p in prev_joint_accelerations]
        curr_joint_positions_rel_obs = normalize_joint_values(curr_joint_position, self._robot_scene.joint_upper_limits)
        curr_joint_velocity_rel_obs = normalize_joint_values(curr_joint_velocity, self._robot_scene.max_velocities)
        curr_joint_acceleration_rel_obs = normalize_joint_values(curr_joint_acceleration,
                                                                 self._robot_scene.max_accelerations)

        # target point for reaching tasks
        target_point_rel_obs = []
        if self._robot_scene.obstacle_wrapper.use_target_points:
            # the function needs to be called even if the return value is not used.
            # Otherwise new target points are not generated
            target_point_pos, target_point_relative_pos, _, target_point_active_obs = \
                self._robot_scene.obstacle_wrapper.get_target_point_observation(
                    compute_relative_pos_norm=self._obs_add_target_point_relative_pos,
                    compute_target_point_joint_pos_norm=False)
            if self._obs_add_target_point_pos:
                target_point_rel_obs = target_point_rel_obs + list(target_point_pos)

            if self._obs_add_target_point_relative_pos:
                target_point_rel_obs = target_point_rel_obs + list(target_point_relative_pos)

            target_point_rel_obs = target_point_rel_obs + list(target_point_active_obs)
            # to indicate if the target point is active (1.0) or inactive (0.0); the list is empty if not required

        observation = list(np.clip([item for sublist in prev_joint_accelerations_rel for item in sublist]
                                   + curr_joint_positions_rel_obs + curr_joint_velocity_rel_obs
                                   + curr_joint_acceleration_rel_obs + target_point_rel_obs, -1, 1))

        info = {'average': {},
                'max': {},
                'min': {}}

        pos_violation = 0.0
        vel_violation = 0.0
        acc_violation = 0.0

        for j in range(self._num_manip_joints):

            info['average']['joint_{}_pos'.format(j)] = curr_joint_positions_rel_obs[j]
            info['average']['joint_{}_pos_abs'.format(j)] = abs(curr_joint_positions_rel_obs[j])
            info['max']['joint_{}_pos'.format(j)] = curr_joint_positions_rel_obs[j]
            info['min']['joint_{}_pos'.format(j)] = curr_joint_positions_rel_obs[j]
            if abs(curr_joint_positions_rel_obs[j]) > 1.001:
                pos_violation = 1.0

            info['average']['joint_{}_vel'.format(j)] = curr_joint_velocity_rel_obs[j]
            info['average']['joint_{}_vel_abs'.format(j)] = abs(curr_joint_velocity_rel_obs[j])
            info['max']['joint_{}_vel'.format(j)] = curr_joint_velocity_rel_obs[j]
            info['min']['joint_{}_vel'.format(j)] = curr_joint_velocity_rel_obs[j]
            if abs(curr_joint_velocity_rel_obs[j]) > 1.001:
                vel_violation = 1.0

            info['average']['joint_{}_acc'.format(j)] = curr_joint_acceleration_rel_obs[j]
            info['average']['joint_{}_acc_abs'.format(j)] = abs(curr_joint_acceleration_rel_obs[j])
            info['max']['joint_{}_acc'.format(j)] = curr_joint_acceleration_rel_obs[j]
            info['min']['joint_{}_acc'.format(j)] = curr_joint_acceleration_rel_obs[j]
            if abs(curr_joint_acceleration_rel_obs[j]) > 1.001:
                acc_violation = 1.0

        info['max']['joint_pos_violation'] = pos_violation
        info['max']['joint_vel_violation'] = vel_violation
        info['max']['joint_acc_violation'] = acc_violation

        return observation, info

    def _get_m_prev_joint_values(self, m, key):

        m_prev_joint_values = []

        for i in range(m+1, 1, -1):
            m_prev_joint_values.append(self._get_generated_trajectory_point(-i, key))

        return m_prev_joint_values
