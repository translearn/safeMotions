# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
from abc import ABC

import numpy as np
from gym.spaces import Box
from klimits import PosVelJerkLimitation
from klimits import denormalize as denormalize
from klimits import get_num_threads
from klimits import interpolate_acceleration_batch as interpolate_acceleration_batch
from klimits import interpolate_position_batch as interpolate_position_batch
from klimits import interpolate_velocity_batch as interpolate_velocity_batch
from klimits import normalize_batch as normalize_batch

from safemotions.envs.safe_motions_base import SafeMotionsBase
from safemotions.utils import trajectory_plotter
from safemotions.utils.braking_trajectory_generator import BrakingTrajectoryGenerator

TERMINATION_JOINT_LIMITS = 1


class AccelerationPredictionBoundedJerkAccVelPos(ABC, SafeMotionsBase):

    def __init__(self,
                 *vargs,
                 limit_velocity=True,
                 limit_position=True,
                 set_velocity_after_max_pos_to_zero=True,
                 acc_limit_factor_braking=1.0,
                 jerk_limit_factor_braking=1.0,
                 plot_trajectory=False,
                 save_trajectory_plot=False,
                 plot_joint=None,
                 plot_acc_limits=False,
                 plot_actual_values=False,
                 plot_time_limits=None,
                 **kwargs):
        super().__init__(*vargs, **kwargs)

        self.action_space = Box(low=np.float32(-1), high=np.float32(1), shape=(self._num_manip_joints,),
                                dtype=np.float32)

        self._plot_trajectory = plot_trajectory
        self._save_trajectory_plot = save_trajectory_plot
        self._limit_velocity = limit_velocity
        self._limit_position = limit_position
        self._safe_acc_range = None

        self._pos_limits_min_max = np.array([self._robot_scene.joint_lower_limits,
                                             self._robot_scene.joint_upper_limits])  # [min_max][joint]
        pos_limits_joint = np.swapaxes(self._pos_limits_min_max, 0, 1)  # [joint][min_max]
        self._vel_limits_min_max = np.array([-1 * self._robot_scene.max_velocities, self._robot_scene.max_velocities])
        vel_limits_joint = np.swapaxes(self._vel_limits_min_max, 0, 1)
        self._acc_limits_min_max = np.array([-1 * self._robot_scene.max_accelerations,
                                             self._robot_scene.max_accelerations])
        acc_limits_joint = np.swapaxes(self._acc_limits_min_max, 0, 1)
        jerk_limits_joint = np.swapaxes(np.array([-1 * self._robot_scene.max_jerk_linear_interpolation,
                                                  self._robot_scene.max_jerk_linear_interpolation]), 0, 1)
        torque_limits_joint = np.swapaxes(np.array([-1 * self._robot_scene.max_torques,
                                                    self._robot_scene.max_torques]), 0, 1)

        max_accelerations_braking = acc_limit_factor_braking * self._robot_scene.max_accelerations
        acc_limits_braking = np.swapaxes(np.array([(-1) * max_accelerations_braking, max_accelerations_braking]), 0, 1)
        max_jerk_braking = np.array([min(2 * jerk_limit_factor_braking * max_accelerations_braking[i]
                                         / self._trajectory_time_step,
                                         self._robot_scene.max_jerk_linear_interpolation[i])
                                     for i in range(len(max_accelerations_braking))])
        jerk_limits_braking = np.swapaxes(np.array([-1 * max_jerk_braking, max_jerk_braking]), 0, 1)

        self._plot_acc_limits = plot_acc_limits
        self._plot_actual_values = plot_actual_values

        if self._plot_trajectory and self._plot_actual_values and self._use_real_robot:
            raise NotImplementedError("Simultaneous plotting of actual values not implemented for real robots")

        num_threads = None
        if num_threads is None:
            logging.warning("Using %s thread(s) per worker to compute the range of safe accelerations.",
                            get_num_threads())
        else:
            logging.warning("Using %s thread(s) per worker to compute the range of safe accelerations.", num_threads)

        self._acc_limitation = PosVelJerkLimitation(time_step=self._trajectory_time_step,
                                                    pos_limits=pos_limits_joint, vel_limits=vel_limits_joint,
                                                    acc_limits=acc_limits_joint, jerk_limits=jerk_limits_joint,
                                                    acceleration_after_max_vel_limit_factor=
                                                    self._acceleration_after_max_vel_limit_factor,
                                                    set_velocity_after_max_pos_to_zero=
                                                    set_velocity_after_max_pos_to_zero,
                                                    limit_velocity=limit_velocity, limit_position=limit_position,
                                                    normalize_acc_range=False,
                                                    num_threads=num_threads)

        self._braking_trajectory_generator = BrakingTrajectoryGenerator(trajectory_time_step=
                                                                        self._trajectory_time_step,
                                                                        acc_limits_braking=acc_limits_braking,
                                                                        jerk_limits_braking=jerk_limits_braking)

        if self._plot_trajectory or self._save_trajectory_plot:
            self._trajectory_plotter = \
                trajectory_plotter.TrajectoryPlotter(time_step=self._trajectory_time_step,
                                                     control_time_step=self._control_time_step,
                                                     computed_actual_values_time_step=self._simulation_time_step,
                                                     pos_limits=pos_limits_joint, vel_limits=vel_limits_joint,
                                                     acc_limits=acc_limits_joint,
                                                     jerk_limits=jerk_limits_joint,
                                                     torque_limits=torque_limits_joint,
                                                     plot_joint=plot_joint,
                                                     plot_acc_limits=self._plot_acc_limits,
                                                     plot_time_limits=plot_time_limits,
                                                     plot_actual_values=self._plot_actual_values,
                                                     plot_computed_actual_values=self._plot_computed_actual_values,
                                                     plot_actual_torques=self._plot_actual_torques)

    def _get_safe_acc_range(self):
        return self._safe_acc_range

    @property
    def pos_limits_min_max(self):
        return self._pos_limits_min_max

    def _reset_plotter(self, initial_joint_position):
        if self._plot_trajectory or self._save_trajectory_plot:
            self._trajectory_plotter.reset_plotter(initial_joint_position)

    def _display_plot(self):
        if self._plot_trajectory:
            self._trajectory_plotter.display_plot(obstacle_wrapper=self._robot_scene.obstacle_wrapper)

    def _add_actual_position_to_plot(self):
        if (self._plot_trajectory and self._plot_actual_values) or self._save_trajectory_plot:
            actual_joint_position = self._robot_scene.get_actual_joint_positions()
            self._trajectory_plotter.add_actual_position(actual_joint_position)

    def _add_computed_actual_position_to_plot(self, computed_position_is, computed_velocity_is,
                                              computed_acceleration_is):
        if self._plot_trajectory and self._plot_computed_actual_values:
            self._trajectory_plotter.add_computed_actual_value(computed_position_is, computed_velocity_is,
                                                               computed_acceleration_is)

    def _add_actual_torques_to_plot(self, actual_torques):
        if self._plot_trajectory and self._plot_actual_torques:
            self._trajectory_plotter.add_actual_torque(actual_torques)

    def _save_plot(self, class_name, experiment_name):
        if self._save_trajectory_plot or (self._log_obstacle_data and self._save_obstacle_data):
            self._trajectory_plotter.save_trajectory(class_name, experiment_name)
        if self._log_obstacle_data and self._save_obstacle_data:
            self._trajectory_plotter.save_obstacle_data(class_name, experiment_name)

    def _calculate_safe_acc_range(self, start_position, start_velocity, start_acceleration, trajectory_point_index):
        # the acc range is required to compute the corresponding mapping to meet the next reference acceleration
        # which can be included into the state. -> Acc range for observation 0 required; called in base reset()
        self._safe_acc_range, _ = self._acc_limitation.calculate_valid_acceleration_range(start_position,
                                                                                          start_velocity,
                                                                                          start_acceleration)

    def compute_next_acc_min_and_next_acc_max(self, start_position, start_velocity, start_acceleration):
        safe_acc_range_joint, _ = self._acc_limitation.calculate_valid_acceleration_range(start_position,
                                                                                          start_velocity,
                                                                                          start_acceleration)

        safe_acc_range_min_max = safe_acc_range_joint.T
        # computes the denormalized minimum and maximum acceleration that can be reached at the following time step
        next_acc_min = safe_acc_range_min_max[0]
        next_acc_max = safe_acc_range_min_max[1]

        return next_acc_min, next_acc_max

    def acc_braking_function(self, *vargs, **kwargs):
        return self._braking_trajectory_generator.get_clipped_braking_acceleration(*vargs, **kwargs)

    def _compute_controller_setpoints_from_action(self, action):
        info = {'average': {},
                'max': {}}

        robot_stopped = False

        safe_acc_range_min_max = self._safe_acc_range.T
        self._end_acceleration = denormalize(action, safe_acc_range_min_max)

        execute_braking_trajectory = False
        self._adaptation_punishment = 0

        if not self._brake:
            self._end_acceleration, execute_braking_trajectory, self._adaptation_punishment, \
                min_distance, max_torque = \
                self._robot_scene.obstacle_wrapper.adapt_action(
                    current_acc=self._start_acceleration,
                    current_vel=self._start_velocity,
                    current_pos=self._start_position,
                    target_acc=self._end_acceleration,
                    time_step_counter=self._current_trajectory_point_index)

            if execute_braking_trajectory:
                self._end_acceleration, robot_stopped, min_distance, max_torque = \
                    self._robot_scene.obstacle_wrapper.get_braking_acceleration()
                if min_distance is not None:
                    self._end_min_distance = min_distance
                if max_torque is not None:
                    self._end_max_torque = max_torque
            else:
                self._end_min_distance = min_distance
                self._end_max_torque = max_torque
        else:
            self._end_acceleration, robot_stopped, min_distance, max_torque = \
                self._robot_scene.obstacle_wrapper.get_braking_acceleration()
            if min_distance is not None:
                self._end_min_distance = min_distance
            if max_torque is not None:
                self._end_max_torque = max_torque

        # compute setpoints
        controller_setpoints, joint_limit_violation = self._compute_interpolated_setpoints()

        if joint_limit_violation and not self._brake:
            self._network_prediction_part_done = True
            self._termination_reason = TERMINATION_JOINT_LIMITS
            self._trajectory_successful = False
            self._brake = True

            if not execute_braking_trajectory:
                self._end_acceleration, robot_stopped, min_distance, max_torque = \
                    self._robot_scene.obstacle_wrapper.get_braking_acceleration(last=True)
                if min_distance is not None:
                    self._end_min_distance = min_distance
                if max_torque is not None:
                    self._end_max_torque = max_torque
                controller_setpoints, _ = self._compute_interpolated_setpoints()

        if self._control_time_step != self._simulation_time_step:
            obstacle_client_update_setpoints, _ = self._compute_interpolated_setpoints(
                use_obstacle_client_update_time_step=True)
        else:
            obstacle_client_update_setpoints = controller_setpoints

        if self._plot_trajectory or self._save_trajectory_plot:
            self._trajectory_plotter.add_data_point(self._end_acceleration, self._safe_acc_range)

        return self._end_acceleration, controller_setpoints, obstacle_client_update_setpoints, info, robot_stopped

    def _compute_interpolated_setpoints(self, use_obstacle_client_update_time_step=False):
        joint_limit_violation = False

        if not use_obstacle_client_update_time_step:
            steps_per_action = self._control_steps_per_action
        else:
            steps_per_action = self._obstacle_client_update_steps_per_action

        time_since_start = np.linspace(self._trajectory_time_step / steps_per_action,
                                       self._trajectory_time_step, steps_per_action)

        interpolated_position_setpoints = interpolate_position_batch(self._start_acceleration,
                                                                     self._end_acceleration,
                                                                     self._start_velocity,
                                                                     self._start_position, time_since_start,
                                                                     self._trajectory_time_step)

        interpolated_velocity_setpoints = interpolate_velocity_batch(self._start_acceleration,
                                                                     self._end_acceleration,
                                                                     self._start_velocity,
                                                                     time_since_start,
                                                                     self._trajectory_time_step)

        interpolated_acceleration_setpoints = interpolate_acceleration_batch(self._start_acceleration,
                                                                             self._end_acceleration,
                                                                             time_since_start,
                                                                             self._trajectory_time_step)

        interpolated_setpoints = {'positions': interpolated_position_setpoints,
                                  'velocities': interpolated_velocity_setpoints,
                                  'accelerations': interpolated_acceleration_setpoints}

        if self._use_real_robot and not self._brake and not use_obstacle_client_update_time_step:

            # Note: It might be more efficient to calculate the min / max absolute value for each joint individually
            # rather than normalizing all values
            max_normalized_position = np.max(np.abs(normalize_batch(interpolated_position_setpoints,
                                                                    self._pos_limits_min_max)))
            max_normalized_velocity = np.max(np.abs(normalize_batch(interpolated_velocity_setpoints,
                                                                    self._vel_limits_min_max)))
            max_normalized_acceleration = np.max(np.abs(normalize_batch(interpolated_acceleration_setpoints,
                                                                        self._acc_limits_min_max)))

            if max_normalized_position > 1.002 or max_normalized_velocity > 1.002 \
                    or max_normalized_acceleration > 1.002:
                joint_limit_violation = True

                if max_normalized_position > 1.002:
                    logging.warning("Position limit exceeded: %s", max_normalized_position)
                if max_normalized_velocity > 1.002:
                    logging.warning("Velocity limit exceeded: %s", max_normalized_velocity)
                if max_normalized_acceleration > 1.002:
                    logging.warning("Acceleration limit exceeded: %s", max_normalized_acceleration)

        return interpolated_setpoints, joint_limit_violation

    def _interpolate_position(self, step):
        interpolated_position = self._start_position + self._start_velocity * step + \
                                0.5 * self._start_acceleration * step ** 2 + \
                                1 / 6 * ((self._end_acceleration - self._start_acceleration)
                                         / self._trajectory_time_step) * step ** 3
        return list(interpolated_position)

    def _interpolate_velocity(self, step):
        interpolated_velocity = self._start_velocity + self._start_acceleration * step + \
                                0.5 * ((self._end_acceleration - self._start_acceleration) /
                                       self._trajectory_time_step) * step ** 2

        return list(interpolated_velocity)

    def _interpolate_acceleration(self, step):
        interpolated_acceleration = self._start_acceleration + \
                                    ((self._end_acceleration - self._start_acceleration) /
                                     self._trajectory_time_step) * step

        return list(interpolated_acceleration)

    def _integrate_linear(self, start_value, end_value):
        return (end_value + start_value) * self._trajectory_time_step / 2

    def _normalize(self, value, value_range):
        normalized_value = -1 + 2 * (value - value_range[0]) / (value_range[1] - value_range[0])
        return normalized_value

    def _denormalize(self, norm_value, value_range):
        actual_value = value_range[0] + 0.5 * (norm_value + 1) * (value_range[1] - value_range[0])
        return actual_value
