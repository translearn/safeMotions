# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import logging
import os
import time
from abc import abstractmethod
from functools import partial
from pathlib import Path
from threading import Thread

import gym
import numpy as np
import pybullet as p

from safemotions.robot_scene.real_robot_scene import RealRobotScene
from safemotions.robot_scene.simulated_robot_scene import SimRobotScene
from safemotions.utils.trajectory_manager import TrajectoryManager

SIM_TIME_STEP = 1. / 240.
CONTROLLER_TIME_STEP = 1. / 200.
EPISODES_PER_SIMULATION_RESET = 50000  # to avoid out of memory error

# Termination reason
TERMINATION_UNSET = -1
TERMINATION_SUCCESS = 0
TERMINATION_JOINT_LIMITS = 1
TERMINATION_TRAJECTORY_LENGTH = 2


class SafeMotionsBase(gym.Env):
    def __init__(self,
                 evaluation_dir=None,
                 use_real_robot=False,
                 real_robot_debug_mode=False,
                 use_gui=False,
                 control_time_step=None,
                 use_control_rate_sleep=True,
                 use_thread_for_control_rate_sleep=False,
                 pos_limit_factor=1,
                 vel_limit_factor=1,
                 acc_limit_factor=1,
                 jerk_limit_factor=1,
                 torque_limit_factor=1,
                 acceleration_after_max_vel_limit_factor=0.01,
                 eval_new_condition_counter=1,
                 log_obstacle_data=False,
                 save_obstacle_data=False,
                 store_actions=False,
                 online_trajectory_duration=8.0,
                 online_trajectory_time_step=0.1,
                 position_controller_time_constants=None,
                 plot_computed_actual_values=False,
                 plot_actual_torques=False,
                 robot_scene=0,
                 obstacle_scene=0,
                 observed_link_point_scene=0,
                 obstacle_use_computed_actual_values=False,
                 visualize_bounding_spheres=False,
                 use_braking_trajectory_method=False,
                 collision_check_time=0.05,
                 check_braking_trajectory_observed_points=False,
                 check_braking_trajectory_closest_points=True,
                 check_braking_trajectory_torque_limits=False,
                 closest_point_safety_distance=0.1,
                 observed_point_safety_distance=0.1,
                 use_target_points=False,
                 target_point_cartesian_range_scene=0,
                 target_point_radius=0.05,
                 target_point_sequence=0,
                 target_point_reached_reward_bonus=0.0,
                 target_link_name="iiwa_link_7",
                 target_link_offset=None,
                 no_self_collision=False,
                 time_step_fraction_sleep_observation=0.0,
                 seed=None,
                 random_agent=False,
                 **kwargs):

        if target_link_offset is None:
            target_link_offset = [0, 0, 0.0]
        if position_controller_time_constants is None:
            position_controller_time_constants = [0.030, 0.030, 0.030, 0.030, 0.030, 0.030, 0.030]
        if seed is not None:
            np.random.seed(seed)
        if evaluation_dir is None:
            self._evaluation_dir = os.path.join(Path.home(), "safe_motions_evaluation")
        else:
            self._evaluation_dir = os.path.join(evaluation_dir, "safe_motions_evaluation")

        self._use_real_robot = use_real_robot
        self._use_gui = use_gui
        self._use_control_rate_sleep = use_control_rate_sleep
        self._num_physic_clients = 0

        if self._use_gui:
            self._simulation_client_id = p.connect(p.GUI)
            self._num_physic_clients += 1
        else:
            if not self._use_real_robot:
                self._simulation_client_id = p.connect(p.DIRECT)
                self._num_physic_clients += 1
            else:
                self._simulation_client_id = None

        self._obstacle_client_id = p.connect(p.DIRECT)
        self._num_physic_clients += 1

        if control_time_step is None:
            self._control_time_step = CONTROLLER_TIME_STEP if self._use_real_robot else SIM_TIME_STEP
        else:
            self._control_time_step = control_time_step

        self._simulation_time_step = SIM_TIME_STEP
        self._control_step_counter = 0
        self._episode_counter = 0

        self._obstacle_scene = obstacle_scene
        self._observed_link_point_scene = observed_link_point_scene
        self._visualize_bounding_spheres = visualize_bounding_spheres
        self._log_obstacle_data = log_obstacle_data
        self._save_obstacle_data = save_obstacle_data
        self._robot_scene_config = robot_scene
        self._use_braking_trajectory_method = use_braking_trajectory_method
        self._collision_check_time = collision_check_time
        self._check_braking_trajectory_observed_points = check_braking_trajectory_observed_points
        self._check_braking_trajectory_closest_points = check_braking_trajectory_closest_points
        self._check_braking_trajectory_torque_limits = check_braking_trajectory_torque_limits
        self._closest_point_safety_distance = closest_point_safety_distance
        self._observed_point_safety_distance = observed_point_safety_distance
        self._use_target_points = use_target_points
        self._target_point_cartesian_range_scene = target_point_cartesian_range_scene
        self._target_point_radius = target_point_radius
        self._target_point_sequence = target_point_sequence
        self._target_point_reached_reward_bonus = target_point_reached_reward_bonus
        self._no_self_collision = no_self_collision
        self._trajectory_time_step = online_trajectory_time_step
        self._position_controller_time_constants = position_controller_time_constants
        self._plot_computed_actual_values = plot_computed_actual_values
        self._plot_actual_torques = plot_actual_torques
        self._pos_limit_factor = pos_limit_factor
        self._vel_limit_factor = vel_limit_factor
        self._acc_limit_factor = acc_limit_factor
        self._jerk_limit_factor = jerk_limit_factor
        self._torque_limit_factor = torque_limit_factor
        self._acceleration_after_max_vel_limit_factor = acceleration_after_max_vel_limit_factor
        self._online_trajectory_duration = online_trajectory_duration
        self._eval_new_condition_counter = eval_new_condition_counter
        self._store_actions = store_actions
        self._target_link_name = target_link_name
        self._target_link_offset = target_link_offset
        self._real_robot_debug_mode = real_robot_debug_mode
        self._random_agent = random_agent

        self._network_prediction_part_done = None
        self._use_thread_for_control_rate_sleep = use_thread_for_control_rate_sleep
        if self._use_thread_for_control_rate_sleep and not self._use_control_rate_sleep:
            logging.warning("use_thread_for_control_rate_sleep ignored since use_control_rate_sleep == False")
        self._use_movement_thread = (self._use_control_rate_sleep and self._use_thread_for_control_rate_sleep) \
                                    or self._use_real_robot
        self._time_step_fraction_sleep_observation = time_step_fraction_sleep_observation
        # 0..1; fraction of the time step,  the main thread sleeps before getting the next observation;
        # only relevant if self._use_real_robot == True
        if time_step_fraction_sleep_observation != 0:
            logging.info("time_step_fraction_sleep_observation %s", self._time_step_fraction_sleep_observation)
        self._obstacle_use_computed_actual_values = obstacle_use_computed_actual_values
        # use computed actual values to determine the distance between the robot and obstacles and as initial point
        # for torque simulations -> advantage: can be computed in advance, no measurements -> real-time capable
        # disadvantage: controller model might introduce inaccuracies
        if self._use_movement_thread and not self._obstacle_use_computed_actual_values:
            raise ValueError("Real-time execution requires obstacle_use_computed_actual_values to be True")

        if self._use_movement_thread:
            logging.info("Using movement thread")

        self._model_actual_values = self._use_movement_thread or self._obstacle_use_computed_actual_values or \
                                    self._plot_computed_actual_values or (self._use_real_robot and self._use_gui)

        if not self._use_movement_thread and self._control_time_step != self._simulation_time_step:
            raise ValueError("If no movement thread is used, the control time step must equal the control time step "
                             "of the obstacle client")

        self._start_position = None
        self._start_velocity = None
        self._start_acceleration = None
        self._position_deviation = None
        self._acceleration_deviation = None
        self._current_trajectory_point_index = None
        self._trajectory_successful = None
        self._total_reward = None
        self._episode_length = None
        self._action_list = []
        self._last_action = None
        self._termination_reason = TERMINATION_UNSET
        self._movement_thread = None
        self._braking = False

        self._init_simulation()

    def _init_simulation(self):
        # reset the physics engine
        for i in range(self._num_physic_clients):
            p.resetSimulation(physicsClientId=i)  # to free memory

        # robot scene settings
        robot_scene_parameters = {'simulation_client_id': self._simulation_client_id,
                                  'simulation_time_step': self._simulation_time_step,
                                  'obstacle_client_id': self._obstacle_client_id,
                                  'trajectory_time_step': self._trajectory_time_step,
                                  'use_real_robot': self._use_real_robot,
                                  'robot_scene': self._robot_scene_config,
                                  'obstacle_scene': self._obstacle_scene,
                                  'observed_link_point_scene': self._observed_link_point_scene,
                                  'log_obstacle_data': self._log_obstacle_data,
                                  'visualize_bounding_spheres': self._visualize_bounding_spheres,
                                  'use_braking_trajectory_method': self._use_braking_trajectory_method,
                                  'collision_check_time': self._collision_check_time,
                                  'check_braking_trajectory_observed_points':
                                      self._check_braking_trajectory_observed_points,
                                  'check_braking_trajectory_closest_points':
                                      self._check_braking_trajectory_closest_points,
                                  'check_braking_trajectory_torque_limits':
                                      self._check_braking_trajectory_torque_limits,
                                  'closest_point_safety_distance': self._closest_point_safety_distance,
                                  'observed_point_safety_distance': self._observed_point_safety_distance,
                                  'use_target_points': self._use_target_points,
                                  'target_point_cartesian_range_scene': self._target_point_cartesian_range_scene,
                                  'target_point_radius': self._target_point_radius,
                                  'target_point_sequence': self._target_point_sequence,
                                  'target_point_reached_reward_bonus': self._target_point_reached_reward_bonus,
                                  'no_self_collision': self._no_self_collision,
                                  'target_link_name': self._target_link_name,
                                  'target_link_offset': self._target_link_offset,
                                  'pos_limit_factor': self._pos_limit_factor,
                                  'vel_limit_factor': self._vel_limit_factor,
                                  'acc_limit_factor': self._acc_limit_factor,
                                  'jerk_limit_factor': self._jerk_limit_factor,
                                  'torque_limit_factor': self._torque_limit_factor
                                  }

        if self._use_real_robot:
            self._robot_scene = RealRobotScene(real_robot_debug_mode=self._real_robot_debug_mode,
                                               **robot_scene_parameters)
        else:
            self._robot_scene = SimRobotScene(**robot_scene_parameters)

        self._num_manip_joints = self._robot_scene.num_manip_joints
        # trajectory manager settings
        self._trajectory_manager = TrajectoryManager(trajectory_time_step=self._trajectory_time_step,
                                                     trajectory_duration=self._online_trajectory_duration,
                                                     obstacle_wrapper=self._robot_scene.obstacle_wrapper)

        self._robot_scene.compute_actual_joint_limits()
        self._control_steps_per_action = int(round(self._trajectory_time_step / self._control_time_step))
        self._obstacle_client_update_steps_per_action = int(round(self._trajectory_time_step /
                                                                  self._simulation_time_step))

        logging.info("Trajectory time step: " + str(self._trajectory_time_step))

        # calculate model coefficients to estimate actual values if required
        if self._model_actual_values:
            self._trajectory_manager.compute_controller_model_coefficients(self._position_controller_time_constants,
                                                                           self._simulation_time_step)

        self._zero_joint_vector = [0.0] * self._num_manip_joints

        if self._use_movement_thread or (self._use_gui and self._use_control_rate_sleep):
            try:
                import rospy
                rospy.init_node("safe_motions_control_rate", anonymous=True, disable_signals=True)
                self._control_rate = rospy.Rate(1. / self._control_time_step)
            except ImportError as _:
                logging.warning("Could not find rospy / a ROS installation. Using time.sleep instead of rospy.Rate. "
                                "Timing will be less accurate.")
                control_rate_sleep = partial(time.sleep, self._control_time_step)
                self._control_rate = type("control_rate", (object, ), {"sleep": lambda: control_rate_sleep()})
        else:
            self._control_rate = None

        for i in range(self._num_physic_clients):
            p.setGravity(0, 0, -9.81, physicsClientId=i)
            p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=i)
            p.setTimeStep(self._simulation_time_step, physicsClientId=i)

    def reset(self):

        self._episode_counter += 1

        if self._episode_counter % EPISODES_PER_SIMULATION_RESET == 0:
            self._init_simulation()

        self._control_step_counter = 0

        self._total_reward = 0
        self._episode_length = 0
        self._trajectory_successful = True
        self._current_trajectory_point_index = 0
        self._action_list = []

        self._network_prediction_part_done = False

        get_new_setup = (((self._episode_counter-1) % self._eval_new_condition_counter) == 0)

        self._robot_scene.obstacle_wrapper.reset_obstacles()
        self._trajectory_manager.reset(get_new_trajectory=get_new_setup)
        self._start_position = np.array(self._get_trajectory_start_position())
        self._start_velocity = np.array(self._zero_joint_vector)
        self._start_acceleration = np.array(self._zero_joint_vector)
        if self._use_real_robot:
            logging.info("Starting position: %s", self._start_position)
        else:
            logging.debug("Starting position: %s", self._start_position)
        self._robot_scene.pose_manipulator(self._start_position)
        self._robot_scene.obstacle_wrapper.reset(start_position=self._start_position)
        self._robot_scene.obstacle_wrapper.update(target_position=self._start_position,
                                                  target_velocity=self._start_velocity,
                                                  target_acceleration=self._start_acceleration,
                                                  actual_position=self._start_position,
                                                  actual_velocity=self._start_velocity)

        self._reset_plotter(self._start_position)
        self._add_computed_actual_position_to_plot(self._start_position, self._zero_joint_vector,
                                                   self._zero_joint_vector)

        if self._plot_actual_torques and not self._use_real_robot:
            # the initial torques are not zero due to gravity
            self._robot_scene.set_motor_control(target_positions=self._start_position,
                                                physics_client_id=self._simulation_client_id)
            p.stepSimulation(physicsClientId=self._simulation_client_id)
            actual_joint_torques = self._robot_scene.get_actual_joint_torques()
            self._add_actual_torques_to_plot(actual_joint_torques)
        else:
            self._add_actual_torques_to_plot(self._zero_joint_vector)

        self._calculate_norm_acc_range(self._start_position, self._start_velocity, self._start_acceleration,
                                       self._current_trajectory_point_index)

        self._termination_reason = TERMINATION_UNSET
        self._last_action = None
        self._network_prediction_part_done = False
        self._movement_thread = None
        self._braking = False

        if self._control_rate is not None:
            # reset control rate timer
            self._control_rate.sleep()

        return None

    def step(self, action):
        self._episode_length += 1

        if self._random_agent:
            action = np.random.uniform(-1, 1, self.action_space.shape)
            # overwrite the desired action with a random action

        if self._store_actions:
            self._action_list.append(action)

        controller_setpoints, obstacle_client_update_setpoints, action_info, robot_stopped = \
            self._compute_controller_setpoints_from_action(action)

        for i in range(len(controller_setpoints['positions'])):
            self._add_generated_trajectory_control_point(controller_setpoints['positions'][i],
                                                         controller_setpoints['velocities'][i],
                                                         controller_setpoints['accelerations'][i])

        for i in range(len(obstacle_client_update_setpoints['positions'])):

            if self._model_actual_values:

                last_position_setpoint = self._start_position if i == 0 else obstacle_client_update_setpoints[
                    'positions'][i - 1]
                computed_position_is = self._trajectory_manager.model_position_controller_to_compute_actual_position(
                    current_position_setpoint=obstacle_client_update_setpoints['positions'][i],
                    last_position_setpoint=last_position_setpoint)
                computed_velocity_is = (np.array(computed_position_is) - np.array(
                    self._get_computed_actual_trajectory_control_point(-1))) / self._simulation_time_step
                computed_acceleration_is = (computed_velocity_is - np.array(
                    self._get_computed_actual_trajectory_control_point(-1,
                                                                       key='velocities'))) / self._simulation_time_step

                self._add_computed_actual_trajectory_control_point(list(computed_position_is),
                                                                   list(computed_velocity_is),
                                                                   list(computed_acceleration_is))
                self._add_computed_actual_position_to_plot(computed_position_is, computed_velocity_is,
                                                           computed_acceleration_is)

                if self._robot_scene.obstacle_wrapper is not None:
                    if self._use_movement_thread or self._obstacle_use_computed_actual_values:
                        self._robot_scene.obstacle_wrapper.update(
                            target_position=obstacle_client_update_setpoints['positions'][i],
                            target_velocity=obstacle_client_update_setpoints['velocities'][i],
                            target_acceleration=obstacle_client_update_setpoints['accelerations'][
                                                                       i],
                            actual_position=computed_position_is,
                            actual_velocity=computed_velocity_is)

        if self._use_movement_thread:
            movement_thread = Thread(target=self._execute_robot_movement,
                                     kwargs=dict(controller_setpoints=controller_setpoints))
            if self._movement_thread is not None:
                self._movement_thread.join()
            movement_thread.start()
            self._movement_thread = movement_thread
            movement_info = {}
        else:
            self._movement_thread = None
            movement_info = self._execute_robot_movement(controller_setpoints=controller_setpoints)

        self._start_position = np.array(obstacle_client_update_setpoints['positions'][-1])
        self._start_velocity = np.array(obstacle_client_update_setpoints['velocities'][-1])
        self._start_acceleration = np.array(obstacle_client_update_setpoints['accelerations'][-1])
        self._add_generated_trajectory_point(self._start_position, self._start_velocity, self._start_acceleration)

        self._current_trajectory_point_index += 1
        self._last_action = action  # store the last action for reward calculation

        self._calculate_norm_acc_range(self._start_position, self._start_velocity, self._start_acceleration,
                                       self._current_trajectory_point_index)

        # sleep for a specified part of the time_step before getting the observation
        if self._time_step_fraction_sleep_observation != 0:
            time.sleep(self._trajectory_time_step * self._time_step_fraction_sleep_observation)

        observation, reward, done, info = self._process_action_outcome(movement_info, action_info)

        if not self._network_prediction_part_done:
            self._total_reward += reward
        else:
            done = True

        if done:
            self._network_prediction_part_done = True

        if not self._network_prediction_part_done:
            self._prepare_for_next_action()
        else:
            if not self._use_real_robot or robot_stopped:
                if self._trajectory_successful:
                    self._termination_reason = TERMINATION_SUCCESS

                else:
                    if self._episode_length == self._trajectory_manager.trajectory_length - 1 and \
                            self._termination_reason == TERMINATION_UNSET:
                        self._termination_reason = TERMINATION_TRAJECTORY_LENGTH

                if self._movement_thread is not None:
                    self._movement_thread.join()
                self._robot_scene.prepare_for_end_of_episode()
                self._prepare_for_end_of_episode()
                observation, reward, _, info = self._process_end_of_episode(observation, reward, done, info)

                if self._store_actions:
                    self._store_action_list()

            else:
                self._braking = True  # slow down the robot prior to stopping the episode
                done = False

        return observation, reward, done, dict(info)

    def _execute_robot_movement(self, controller_setpoints):
        # executed in real-time if required
        actual_joint_torques_rel_abs_list = []
        for i in range(len(controller_setpoints['positions'])):
            if not self._use_real_robot:
                curr_position_is = self._robot_scene.get_actual_joint_positions()
                curr_velocity_is = (np.array(curr_position_is) - np.array(
                    self._get_measured_actual_trajectory_control_point(-1))) / self._control_time_step
                curr_acceleration_is = (curr_velocity_is - np.array(
                    self._get_measured_actual_trajectory_control_point(-1, key='velocities'))) / self._control_time_step
                self._add_measured_actual_trajectory_control_point(list(curr_position_is), list(curr_velocity_is),
                                                                   list(curr_acceleration_is))
                self._add_actual_position_to_plot(curr_position_is)

            if self._control_rate is not None:
                self._control_rate.sleep()

            self._robot_scene.set_motor_control(controller_setpoints['positions'][i],
                                                computed_position_is=controller_setpoints['positions'][i],
                                                computed_velocity_is=controller_setpoints['velocities'][i])

            if not self._use_real_robot:
                self._sim_step()
                actual_joint_torques = self._robot_scene.get_actual_joint_torques()
                actual_joint_torques_rel_abs = np.abs(normalize_joint_values(actual_joint_torques,
                                                                             self._robot_scene.max_torques))
                actual_joint_torques_rel_abs_list.append(actual_joint_torques_rel_abs)

                if self._plot_actual_torques:
                    self._add_actual_torques_to_plot(actual_joint_torques)
            else:
                actual_joint_torques_rel_abs_list.append(self._zero_joint_vector)

            if self._robot_scene.obstacle_wrapper is not None:
                if not self._use_movement_thread and not self._obstacle_use_computed_actual_values:
                    actual_position, actual_velocity = self._robot_scene.get_actual_joint_position_and_velocity()

                    self._robot_scene.obstacle_wrapper.update(target_position=controller_setpoints['positions'][i],
                                                              target_velocity=controller_setpoints['velocities'][i],
                                                              target_acceleration=controller_setpoints['accelerations'][
                                                                  i],
                                                              actual_position=actual_position,
                                                              actual_velocity=actual_velocity)

        movement_info = {'average': {}, 'max': {}}

        # add torque info to movement_info
        torque_violation = 0.0
        actual_joint_torques_rel_abs_swap = np.swapaxes(actual_joint_torques_rel_abs_list, 0, 1)
        for j in range(self._num_manip_joints):
            movement_info['average']['joint_{}_torque_abs'.format(j)] = np.mean(actual_joint_torques_rel_abs_swap[j])
            actual_joint_torques_rel_abs_max = np.max(actual_joint_torques_rel_abs_swap[j])
            movement_info['max']['joint_{}_torque_abs'.format(j)] = actual_joint_torques_rel_abs_max
            if actual_joint_torques_rel_abs_max > 1.001:
                torque_violation = 1.0
                logging.warning("Torque Violation: t = " + str(
                    (self._episode_length - 1) * self._trajectory_time_step) + " Joint: " + str(
                    j) + " Rel Torque:" + str(actual_joint_torques_rel_abs_max))

        movement_info['max']['joint_torque_violation'] = torque_violation
        movement_info['average']['joint_torque_violation'] = torque_violation

        return movement_info

    def close(self):
        self._robot_scene.disconnect()

    @abstractmethod
    def render(self, mode="human"):
        raise NotImplementedError()

    @abstractmethod
    def _get_observation(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_reward(self):
        raise NotImplementedError()

    @abstractmethod
    def _compute_controller_setpoints_from_action(self, action):
        raise NotImplementedError()

    @abstractmethod
    def _interpolate_position(self, step):
        raise NotImplementedError()

    @abstractmethod
    def _interpolate_velocity(self, step):
        raise NotImplementedError()

    @abstractmethod
    def _interpolate_acceleration(self, step):
        raise NotImplementedError()

    @abstractmethod
    def _process_action_outcome(self, base_info, action_info):
        raise NotImplementedError()

    @abstractmethod
    def _process_end_of_episode(self, observation, reward, done, info):
        raise NotImplementedError()

    def _sim_step(self):
        p.stepSimulation(physicsClientId=self._simulation_client_id)
        self._control_step_counter += 1

    def _prepare_for_next_action(self):
        return

    def _prepare_for_end_of_episode(self):
        return

    @abstractmethod
    def _store_action_list(self):
        raise NotImplementedError()

    def _check_termination(self):
        done = False
        if self._trajectory_manager.is_trajectory_finished(self._current_trajectory_point_index):
            done = True

        return done

    @property
    def norm_acc_range(self):
        return self._get_norm_acc_range()

    @property
    def trajectory_time_step(self):
        return self._trajectory_time_step

    @abstractmethod
    def _get_norm_acc_range(self):
        pass

    @abstractmethod
    def _reset_plotter(self, initial_joint_position):
        pass

    @abstractmethod
    def _add_actual_position_to_plot(self, actual_joint_position):
        pass

    @abstractmethod
    def _add_computed_actual_position_to_plot(self, computed_position_is, computed_velocity_is,
                                              computed_acceleration_is):
        pass

    @abstractmethod
    def _add_actual_torques_to_plot(self, actual_torques):
        pass

    @abstractmethod
    def _calculate_norm_acc_range(self, start_position, start_velocity, start_acceleration, trajectory_point_index):
        pass

    def _get_trajectory_start_position(self):
        return self._trajectory_manager.get_trajectory_start_position()

    def _get_generated_trajectory_point(self, index, key='positions'):
        return self._trajectory_manager.get_generated_trajectory_point(index, key)

    def _get_measured_actual_trajectory_control_point(self, index, key='positions'):
        return self._trajectory_manager.get_measured_actual_trajectory_control_point(index, key)

    def _get_computed_actual_trajectory_control_point(self, index, key='positions'):
        return self._trajectory_manager.get_computed_actual_trajectory_control_point(index, key)

    def _get_generated_trajectory_control_point(self, index, key='positions'):
        return self._trajectory_manager.get_generated_trajectory_control_point(index, key)

    def _add_generated_trajectory_point(self, position, velocity, acceleration):
        self._trajectory_manager.add_generated_trajectory_point(position, velocity, acceleration)

    def _add_measured_actual_trajectory_control_point(self, position, velocity, acceleration):
        self._trajectory_manager.add_measured_actual_trajectory_control_point(position, velocity, acceleration)

    def _add_computed_actual_trajectory_control_point(self, position, velocity, acceleration):
        self._trajectory_manager.add_computed_actual_trajectory_control_point(position, velocity, acceleration)

    def _add_generated_trajectory_control_point(self, position, velocity, acceleration):
        self._trajectory_manager.add_generated_trajectory_control_point(position, velocity, acceleration)


def normalize_joint_values(values, joint_limits):
    return list(np.array(values) / np.array(joint_limits))
