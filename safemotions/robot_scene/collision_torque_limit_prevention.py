# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import inspect
import itertools
import logging
import os
import re
import warnings

import numpy as np
import pybullet as p
from klimits import compute_distance_c
from klimits import interpolate_position_batch as interpolate_position_batch
from klimits import interpolate_velocity_batch as interpolate_velocity_batch
from klimits import interpolate_velocity as interpolate_velocity
from klimits import normalize

OBSERVED_POINT_NO_INFLUENCE_COLOR = (0, 1, 0, 0.5)  # green
OBSERVED_POINT_INFLUENCE_COLOR = (255 / 255, 84 / 255, 0 / 255, 0.5)  # orange
OBSERVED_POINT_VIOLATION_COLOR = (1, 0, 0, 0.5)  # red
LINK_OBJECT_COLLISION_INFLUENCE_COLOR = (0 / 255, 0 / 255, 170 / 255, 1.0)
LINK_SELF_COLLISION_INFLUENCE_COLOR = (117 / 255, 5 / 255, 45 / 255, 1.0)
# braking trajectory due to potential self collision
LINK_TORQUE_INFLUENCE_COLOR = (1, 0.33, 0.0, 1.0)  # braking trajectory due to potential torque violation

TARGET_POINT_SIMULTANEOUS = 0
TARGET_POINT_ALTERNATING = 1
TARGET_POINT_SINGLE = 2


class ObstacleWrapperBase:
    def __init__(self,
                 robot_scene=None,
                 obstacle_scene=None,
                 visual_mode=False,
                 observed_link_point_scene=0,
                 log_obstacle_data=False,
                 check_braking_trajectory_collisions=False,
                 check_braking_trajectory_torque_limits=False,
                 collision_check_time=None,
                 simulation_time_step=1 / 240,
                 acc_range_function=None,
                 acc_braking_function=None,
                 check_braking_trajectory_observed_points=False,
                 check_braking_trajectory_closest_points=True,
                 closest_point_safety_distance=0.1,
                 observed_point_safety_distance=0.1,
                 print_stats=False,
                 use_target_points=True,
                 target_point_cartesian_range_scene=0,
                 target_point_relative_pos_scene=0,
                 target_point_radius=0.05,
                 target_point_sequence=TARGET_POINT_SIMULTANEOUS,
                 target_point_reached_reward_bonus=0,
                 target_point_use_actual_position=False,
                 # True: Check if a target point is reached based on the actual position, False: Use setpoints
                 reward_maximum_relevant_distance=None,
                 *vargs,
                 **kwargs):

        self._robot_scene = robot_scene
        self._obstacle_scene = obstacle_scene
        self._visual_mode = visual_mode
        self._observed_link_point_scene = observed_link_point_scene
        self._obstacle_list = []
        self._links_in_use = []
        self._links = []
        self._log_obstacle_data = log_obstacle_data
        self._trajectory_time_step = None
        self._simulation_time_step = simulation_time_step

        self._acc_range_function = acc_range_function
        self._acc_braking_function = acc_braking_function
        self._check_braking_trajectory_collisions = check_braking_trajectory_collisions
        self._check_braking_trajectory_torque_limits = check_braking_trajectory_torque_limits
        self._braking_trajectory = None
        self._valid_braking_trajectories = None
        self._use_braking_trajectory_method = self._check_braking_trajectory_collisions \
            or self._check_braking_trajectory_torque_limits
        self._collision_checks_per_time_step = None
        if collision_check_time is None:
            self._collision_check_time = 0.05
        else:
            self._collision_check_time = collision_check_time
        self._check_braking_trajectory_observed_points = check_braking_trajectory_observed_points
        self._check_braking_trajectory_closest_points = check_braking_trajectory_closest_points
        self._closest_point_safety_distance = closest_point_safety_distance
        self._observed_point_safety_distance = observed_point_safety_distance

        if self._check_braking_trajectory_collisions and not self._check_braking_trajectory_closest_points \
                and not self._check_braking_trajectory_observed_points:
            logging.warning("Warning: check_braking_trajectory_collisions is True but neither closest points nor "
                            "observed points are checked for collisions")

        self._print_stats = print_stats

        self._episode_counter = 0
        self._simulation_steps_per_action = None
        self._mean_num_points_in_safety_zone = 0
        self._mean_num_points_in_collision_zone = 0
        self._num_points_in_safety_zone_list = []
        self._num_points_in_collision_zone_list = []
        self._braking_duration_list = []  # duration of all computed braking trajectories
        self._active_braking_duration_list = []  # duration of all braking trajectories that led to action adaption
        self._active_braking_influence_time_list = []
        # for each introduced braking trajectory, the time that the action is influenced
        self._active_braking_influence_time = None

        self._mean_time_in_collision_zone = 0
        self._mean_time_in_safety_zone = 0
        self._mean_time_influenced_by_braking_trajectory_collision = 0
        self._mean_time_influenced_by_braking_trajectory_torque = 0

        self._time_in_object_observed_point_collision_zone_list = []
        self._time_in_object_closest_point_collision_zone_list = []
        self._time_in_self_collision_zone_list = []
        self._time_in_any_collision_zone_list = []
        self._time_in_safety_zone_list = []
        self._time_influenced_by_braking_trajectory_collision_list = []
        self._time_influenced_by_braking_trajectory_torque_list = []

        self._use_target_points = use_target_points
        self._target_point_radius = target_point_radius
        self._target_point_joint_pos_list = None
        self._sample_new_target_point_list = None
        self._target_point_list = [[] for _ in range(self._robot_scene.num_robots)]

        self._target_position = None
        self._target_velocity = None
        self._actual_position = None
        self._actual_velocity = None
        self._last_actual_position = None
        self._last_actual_velocity = None

        if self._robot_scene.robot_name == "iiwa7":
            self._starting_point_cartesian_range = [[-0.6, 0.6], [-0.8, 0.8],
                                                    [0.1, 1]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
            self._target_point_relative_pos_min_max = np.array([[-1.6, -2, -1.5], [1.6, 2, 1.5]])
        elif self._robot_scene.robot_name.startswith("armar6"):
            if self._robot_scene.robot_name == "armar6":
                self._starting_point_cartesian_range = [[-0.1, 0.75], [-1.1, 1.1],
                                                        [0.3, 1.4]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
                if target_point_relative_pos_scene == 0:
                    self._target_point_relative_pos_min_max = np.array([[-2, -2.6, -2.2], [2, 2.6, 2.2]])
                elif target_point_relative_pos_scene == 1:
                    self._target_point_relative_pos_min_max = np.array([[-1.45, -2.22, -2.0], [2.12, 2.22, 1.42]])
                else:
                    self._target_point_relative_pos_min_max = np.array([[-2, -2.6, -2.2], [2, 2.6, 2.2]])
            elif self._robot_scene.robot_name == "armar6_x4":
                self._starting_point_cartesian_range = [[-0.1, 0.75], [-1.1, 1.1],
                                                        [0.15, 1.6]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

                if target_point_relative_pos_scene == 0:
                    self._target_point_relative_pos_min_max = np.array([[-2, -2.6, -2.2], [2, 2.6, 2.2]])
                elif target_point_relative_pos_scene == 1:
                    self._target_point_relative_pos_min_max = np.array([[-1.55, -2.55, -2.3], [2.24, 2.55, 1.64]])
                else:
                    self._target_point_relative_pos_min_max = np.array([[-2, -2.6, -2.2], [2, 2.6, 2.2]])
        elif self._robot_scene.robot_name.startswith("armar4"):
            self._starting_point_cartesian_range = [[-0.1, 0.75], [-0.9, 0.9],
                                                    [1.0, 1.6]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]
            self._target_point_relative_pos_min_max = np.array([[-2, -2.6, -2.2], [2, 2.6, 2.2]])

        if target_point_cartesian_range_scene == 0:
            self._target_point_cartesian_range = [[-0.6, 0.6], [-0.8, 0.8],
                                                  [0.1, 1]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

        elif target_point_cartesian_range_scene == 1:
            self._target_point_cartesian_range = [[-0.6, 0.6], [-0.3, 0.3],
                                                  [0.1, 1]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

        elif target_point_cartesian_range_scene == 2:
            self._target_point_cartesian_range = [[-0.4, 0.4], [-0.4, 0.4],
                                                  [0.1, 1]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

        elif target_point_cartesian_range_scene == 3:
            self._target_point_cartesian_range = [[-0.2, 0.8], [-0.9, 0.9],
                                                  [0.3, 1.45]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        elif target_point_cartesian_range_scene == 4:
            self._target_point_cartesian_range = [[-0.8, 0.8], [-0.9, 0.9],
                                                  [0.3, 1.45]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        elif target_point_cartesian_range_scene == 5:
            self._target_point_cartesian_range = [[-0.1, 0.8], [-0.9, 0.9],
                                                  [0.3, 1.45]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        elif target_point_cartesian_range_scene == 6:
            self._target_point_cartesian_range = [[-0.1, 0.8], [-1.0, 1.0],
                                                  [0.15, 1.65]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        elif target_point_cartesian_range_scene == 7:
            self._target_point_cartesian_range = [[-0.1, 0.75], [-0.8, 0.8],
                                                  [1.0, 1.65]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        else:
            self._target_point_cartesian_range = [[-0.6, 0.6], [-0.8, 0.8],
                                                  [0.1, 1]]  # [[x_min, x_max], [y_min, y_max], [z_min, z_max]]

        self._log_target_point_relative_pos_min_max = True
        # whether to add the maximum target_point_relative_pos to info
        self._target_point_relative_pos_min_max_log = [[np.inf, np.inf, np.inf], [-np.inf, -np.inf, -np.inf]]

        self._target_point_cartesian_range_min_max = np.array(self._target_point_cartesian_range).T
        self._target_link_pos_list = [None] * self._robot_scene.num_robots
        self._target_point_pos_list = [None] * self._robot_scene.num_robots
        self._last_target_point_distance_list = [None] * self._robot_scene.num_robots
        self._target_point_pos_norm_list = [None] * self._robot_scene.num_robots
        self._target_point_joint_pos_norm_list = [None] * self._robot_scene.num_robots
        self._target_point_reached_list = [None] * self._robot_scene.num_robots
        self._initial_target_point_distance_list = [[np.nan] for _ in range(self._robot_scene.num_robots)]
        self._num_target_points_reached_list = [0] * self._robot_scene.num_robots
        self._starting_point_sampling_attempts = 0
        self._target_point_sampling_attempts_list = [[] for _ in range(self._robot_scene.num_robots)]

        self._braking_trajectory_minimum_distance = np.inf
        self._braking_trajectory_maximum_rel_torque = 0

        self._torque_limits = None
        self._braking_timeout = False

        self._target_point_sequence = target_point_sequence
        # 0: target points for all robots, 1: alternating target points
        self._target_point_active_list = [False] * self._robot_scene.num_robots
        self._target_point_reached_reward_bonus = target_point_reached_reward_bonus

        self._target_point_use_actual_position = target_point_use_actual_position
        if reward_maximum_relevant_distance is None:
            self._closest_point_maximum_relevant_distance = self._closest_point_safety_distance + 0.002
            # for performance purposes the exact distance between links / obstacles is only computed if the
            # distance is smaller than self._closest_point_maximum_relevant_distance
        else:
            # set the maximum relevant distance for computing closest points based on the reward settings
            # -> higher values require more computational effort
            if reward_maximum_relevant_distance <= self._closest_point_safety_distance:
                raise ValueError("reward_maximum_relevant_distance {} needs to be greater than "
                                 "closest_point_safety_distance {}".format(reward_maximum_relevant_distance,
                                                                           self._closest_point_safety_distance))
            else:
                self._closest_point_maximum_relevant_distance = reward_maximum_relevant_distance + 0.002


    @property
    def obstacle_scene(self):
        return self._obstacle_scene

    @property
    def use_target_points(self):
        return self._use_target_points

    @property
    def log_obstacle_data(self):
        return self._log_obstacle_data

    @property
    def num_obstacles(self):
        return len(self._obstacle_list)

    @property
    def target_point_sequence(self):
        return self._target_point_sequence

    @property
    def closest_point_safety_distance(self):
        return self._closest_point_safety_distance

    @property
    def num_target_points(self):
        num_target_points = 0
        for i in range(len(self._target_point_list)):
            num_target_points += len(self._target_point_list[i])

        return num_target_points

    def get_num_target_points_reached(self, robot=None):
        # if robot is None -> consider all robots, otherwise only the specified robot_index
        num_target_points_reached = 0
        for i in range(self._robot_scene.num_robots):
            if robot is None or robot == i:
                num_target_points_reached += self._num_target_points_reached_list[i]

        return num_target_points_reached

    def _get_initial_target_point_distance(self, robot=None):
        # if robot is None -> consider all robots, otherwise only the specified robot_index
        if robot is None:
            return np.nanmean(list(itertools.chain.from_iterable(self._initial_target_point_distance_list)))
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                return np.nanmean(self._initial_target_point_distance_list[robot])

    @property
    def obstacle(self):
        return self._obstacle_list + list(itertools.chain.from_iterable(self._target_point_list))

    @property
    def target_point(self):
        return list(itertools.chain.from_iterable(self._target_point_list))

    @property
    def links_in_use(self):
        return self._links_in_use

    @property
    def links(self):
        return self._links

    @property
    def trajectory_time_step(self):
        return self._trajectory_time_step

    @trajectory_time_step.setter
    def trajectory_time_step(self, val):
        self._trajectory_time_step = val
        self._collision_checks_per_time_step = round(max(1, self._trajectory_time_step / self._collision_check_time))
        self._simulation_steps_per_action = int(round(self._trajectory_time_step / self._simulation_time_step))

    @property
    def torque_limits(self):
        return self._torque_limits

    @torque_limits.setter
    def torque_limits(self, val):
        self._torque_limits = val

    def get_indices_of_observed_links_in_use(self, obstacle_index):
        indices_of_observed_links_in_use = []
        if obstacle_index >= self.num_obstacles:
            obstacle_index = obstacle_index - self.num_obstacles
            obstacle_list = list(itertools.chain.from_iterable(self._target_point_list))
        else:
            obstacle_list = self._obstacle_list
        for i in range(len(self._links_in_use)):
            if self._links_in_use[i] in obstacle_list[obstacle_index].observed_links:
                indices_of_observed_links_in_use.append(i)
        return indices_of_observed_links_in_use

    def get_index_of_link_in_use(self, link_number):
        for i in range(len(self._links_in_use)):
            if self._links_in_use[i] == link_number:
                return i
        raise ValueError("Desired link is not in use")


class ObstacleWrapperSim(ObstacleWrapperBase):
    OBSTACLE_CLIENT_AT_OTHER_POSITION = 0
    OBSTACLE_CLIENT_AT_TARGET_POSITION = 1
    OBSTACLE_CLIENT_AT_ACTUAL_POSITION = 2

    def __init__(self,
                 simulation_client_id=None,
                 obstacle_client_id=None,
                 use_real_robot=False,
                 print_link_infos=True,
                 link_name_list=None,
                 manip_joint_indices=None,
                 visualize_bounding_spheres=False,
                 visualize_bounding_sphere_actual=True,
                 target_link_name="iiwa_link_7",
                 target_link_offset=None,
                 activate_obstacle_collisions=False,
                 *vargs,
                 **kwargs):
        super().__init__(*vargs, **kwargs)

        if target_link_offset is None:
            target_link_offset = [0, 0, 0.0]
        if manip_joint_indices is None:
            manip_joint_indices = []

        self._simulation_client_id = simulation_client_id
        if self._log_obstacle_data and self._simulation_client_id is None:
            raise ValueError("log_obstacle_data requires an active physics client (physicsClientId is None)")
        self._obstacle_client_id = obstacle_client_id
        self._obstacle_client_status = self.OBSTACLE_CLIENT_AT_OTHER_POSITION
        self._use_real_robot = use_real_robot

        self._manip_joint_indices = manip_joint_indices

        self._visualize_bounding_spheres = visualize_bounding_spheres
        if self._visualize_bounding_spheres and not self._log_obstacle_data:
            raise ValueError("visualize_bounding_spheres requires log_obstacle_data to be True")
        self._visualize_bounding_sphere_actual = visualize_bounding_sphere_actual
        self._activate_obstacle_collisions = activate_obstacle_collisions

        if self._use_target_points:
            target_link_name_list = self._robot_scene.get_link_names_for_multiple_robots(target_link_name)
            if len(target_link_name_list) != self._robot_scene.num_robots:
                raise ValueError("Could not find a target link for each robot. Found " + str(target_link_name_list))
            self._target_link_point_list = []
            self._target_link_index_list = []

        self._target_link_name = target_link_name

        if self._robot_scene.robot_name == "iiwa7":
            closest_point_active_link_name_list = ["iiwa_link_2", "iiwa_link_3", "iiwa_link_4", "iiwa_link_5",
                                                   "iiwa_link_6", "iiwa_link_7"]

            closest_point_active_link_name_multiple_robots_list = self._robot_scene.get_link_names_for_multiple_robots(
                closest_point_active_link_name_list)
        elif self._robot_scene.robot_name.startswith("armar6"):
            closest_point_active_link_name_list = ["arm_t34", "arm_t45", "arm_t56", "arm_t67",
                                                   "arm_t78", "arm_t8", "hand_fixed"]
            closest_point_active_link_name_multiple_robots_list = self._robot_scene.get_link_names_for_multiple_robots(
                closest_point_active_link_name_list)
        elif self._robot_scene.robot_name == "armar4_fixed_hands_and_legs":
            closest_point_active_link_name_list = ["torso_pitch", "arm_sho4",
                                                   "arm_elb1", "arm_elb2", "arm_wri2"]
            closest_point_active_link_name_multiple_robots_list = self._robot_scene.get_link_names_for_multiple_robots(
                closest_point_active_link_name_list)
        else:
            closest_point_active_link_name_multiple_robots_list = []

        visual_shape_data = p.getVisualShapeData(self._robot_scene.robot_id,
                                                 flags=p.VISUAL_SHAPE_DATA_TEXTURE_UNIQUE_IDS)
        visualize_target_link_point = False  # bounding sphere around the target link point

        for i in range(len(link_name_list)):
            observed_points = self._specify_observed_points(link_name=link_name_list[i], link_index=i)
            self_collision_links = self._specify_self_collision_links(link_name=link_name_list[i],
                                                                      link_name_list=link_name_list)
            if link_name_list[i] in closest_point_active_link_name_multiple_robots_list:
                closest_point_active = True
            else:
                closest_point_active = False

            # link default color
            default_color = visual_shape_data[i][7]

            if self._use_target_points:
                for j in range(self._robot_scene.num_robots):
                    if link_name_list[i] == self._robot_scene.get_link_names_for_multiple_robots(self._target_link_name,
                                                                                                 robot_indices=[j])[0]:
                        robot_index = j if self._target_point_sequence != TARGET_POINT_SINGLE else 0
                        default_color = self.get_target_point_color(robot=robot_index, transparency=1.0)

            self._links.append(
                LinkBase(name=link_name_list[i], observe_closest_point=True, closest_point_active=closest_point_active,
                         observed_points=observed_points, index=i,
                         closest_point_safety_distance=self._closest_point_safety_distance,
                         robot_id=self._robot_scene.robot_id,
                         robot_index=self._robot_scene.get_robot_index_from_link_name(link_name_list[i]),
                         self_collision_links=self_collision_links,
                         default_color=default_color,
                         simulation_client_id=self._simulation_client_id,
                         obstacle_client_id=self._obstacle_client_id,
                         use_real_robot=self._use_real_robot,
                         set_robot_position_in_obstacle_client_function=self.set_robot_position_in_obstacle_client,
                         is_obstacle_client_at_other_position_function=self.is_obstacle_client_at_other_position
                         ))

            if self._use_target_points:
                if link_name_list[i] in target_link_name_list:
                    self._target_link_point_list.append(LinkPointBase(name="Target", offset=target_link_offset,
                                                                      bounding_sphere_radius=0.01
                                                                      if visualize_target_link_point else 0.0,
                                                                      safety_distance=0.0,
                                                                      active=False,
                                                                      visualize_bounding_sphere=
                                                                      visualize_target_link_point,
                                                                      num_clients=self._robot_scene.num_clients))
                    self._target_link_point_list[-1].link_object = self._links[-1]
                    self._target_link_index_list.append(i)

            if print_link_infos:
                dynamics_info = p.getDynamicsInfo(self._robot_scene.robot_id, i)
                logging.debug("Link " + str(i) + " " + link_name_list[i] + " Mass: " + str(dynamics_info[0]) +
                              " COM pos: " + str(dynamics_info[3]) +
                              " COM orn: " + str(p.getEulerFromQuaternion(dynamics_info[4])) +
                              " Inertia diagonal: " + str(dynamics_info[2]))

        if self._use_target_points:
            if len(self._target_link_point_list) != self._robot_scene.num_robots:
                raise ValueError("Could not find a target link named " + self._target_link_name +
                                 " for each robot. Found " + str(self._target_link_point_list))

        # Visualize the distance between an obstacle and a selected point by a debug line
        self._debug_line = None
        self._debug_line_obstacle = 0
        self._debug_line_link = 0
        self._debug_line_point = 0  # 0: closest point if observed, else: first observed point

        visualize_starting_point_cartesian_range = False
        if visualize_starting_point_cartesian_range:
            self._visualize_cartesian_range(c_range=self._starting_point_cartesian_range, line_color=[1, 0, 0])
        visualize_target_point_cartesian_range = False
        if visualize_target_point_cartesian_range:
            self._visualize_cartesian_range(c_range=self._target_point_cartesian_range, line_color=[0, 0, 1])

    def _visualize_cartesian_range(self, c_range, line_color, line_width=2):
        # c_range: [[x_min, x_max], [y_min, y_max], [z_min, z_max]]
        indices = [[0, 0, 0, 1, 0, 0],
                   [0, 0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0, 1],
                   [0, 1, 0, 1, 1, 0],
                   [0, 1, 0, 0, 1, 1],
                   [0, 0, 1, 0, 1, 1],
                   [0, 0, 1, 1, 0, 1],
                   [0, 1, 1, 1, 1, 1],
                   [1, 0, 0, 1, 1, 0],
                   [1, 0, 0, 1, 0, 1],
                   [1, 1, 0, 1, 1, 1],
                   [1, 0, 1, 1, 1, 1]]

        for client in [self._simulation_client_id, self._obstacle_client_id]:
            if client is not None:
                for index in indices:
                    p.addUserDebugLine([c_range[0][index[0]], c_range[1][index[1]], c_range[2][index[2]]],
                                       [c_range[0][index[3]], c_range[1][index[4]], c_range[2][index[5]]],
                                       lineColorRGB=line_color, lineWidth=line_width,
                                       physicsClientId=client)

    def is_obstacle_client_at_other_position(self):
        if self._obstacle_client_status == self.OBSTACLE_CLIENT_AT_OTHER_POSITION:
            return True
        else:
            return False

    def reset_obstacles(self):
        self._delete_all_target_points()
        if not self._obstacle_list:
            self._add_obstacles()
        else:
            for obstacle in self._obstacle_list:
                obstacle.reset()

    def reset(self, start_position):

        self._episode_counter += 1
        self._time_in_safety_zone_list = []
        self._time_in_object_observed_point_collision_zone_list = []
        self._time_in_object_closest_point_collision_zone_list = []
        self._time_in_self_collision_zone_list = []
        self._time_in_any_collision_zone_list = []
        self._num_points_in_safety_zone_list = []
        self._num_points_in_collision_zone_list = []
        self._time_influenced_by_braking_trajectory_collision_list = []
        self._time_influenced_by_braking_trajectory_torque_list = []
        self._braking_duration_list = []
        self._active_braking_duration_list = []
        self._active_braking_influence_time_list = []
        self._active_braking_influence_time = 0

        self._valid_braking_trajectories = {'current': None, 'last': None}
        self._braking_trajectory_minimum_distance = np.inf
        self._braking_trajectory_maximum_rel_torque = 0

        self._braking_timeout = False

        self._target_position = start_position
        self._target_velocity = np.array([0.0] * len(self._target_position))
        self._actual_position = start_position
        self._actual_velocity = np.array([0.0] * len(self._target_position))
        self._last_actual_position = None
        self._last_actual_velocity = None


        self._obstacle_client_status = self.OBSTACLE_CLIENT_AT_OTHER_POSITION

        for i in range(len(self._links)):
            self._links[i].reset()

        if self._use_target_points:
            self._target_point_sampling_attempts_list = [[] for _ in range(self._robot_scene.num_robots)]
            self._last_target_point_distance_list = [None] * self._robot_scene.num_robots
            self._sample_new_target_point_list = [False] * self._robot_scene.num_robots
            self._target_point_pos_norm_list = [None] * self._robot_scene.num_robots
            self._target_point_joint_pos_list = [None] * self._robot_scene.num_robots
            self._target_point_joint_pos_norm_list = [None] * self._robot_scene.num_robots
            self._target_point_active_list = [False] * self._robot_scene.num_robots
            self._target_point_reached_list = [False] * self._robot_scene.num_robots
            self._initial_target_point_distance_list = [[np.nan] for _ in range(self._robot_scene.num_robots)]
            self._num_target_points_reached_list = [0] * self._robot_scene.num_robots

            self._target_point_relative_pos_min_max_log = [[np.inf, np.inf, np.inf], [-np.inf, -np.inf, -np.inf]]

            if self._target_point_sequence == TARGET_POINT_SIMULTANEOUS:
                self._target_point_active_list = [True] * self._robot_scene.num_robots
            else:
                target_point_active_robot_index = np.random.randint(0, self._robot_scene.num_robots)
                self._target_point_active_list[target_point_active_robot_index] = True

            active_robots_list = []
            for i in range(self._robot_scene.num_robots):
                if self._target_point_active_list[i]:
                    active_robots_list.append(i)
            np.random.shuffle(active_robots_list)
            for i in range(len(active_robots_list)):
                self._add_target_point(robot=active_robots_list[i])

    def get_info_and_print_stats(self, print_stats_every_n_episodes=1):
        episode_mean_time_influenced_by_braking_trajectory_collision = np.mean(
            self._time_influenced_by_braking_trajectory_collision_list)
        episode_mean_time_influenced_by_braking_trajectory_torque = np.mean(
            self._time_influenced_by_braking_trajectory_torque_list)
        info = {'obstacles_time_influenced_by_braking_trajectory':
                    episode_mean_time_influenced_by_braking_trajectory_collision +
                    episode_mean_time_influenced_by_braking_trajectory_torque,
                'obstacles_time_influenced_by_braking_trajectory_collision':
                    episode_mean_time_influenced_by_braking_trajectory_collision,
                'obstacles_time_influenced_by_braking_trajectory_torque':
                    episode_mean_time_influenced_by_braking_trajectory_torque,
                'obstacles_num_target_points_reached': float(self.get_num_target_points_reached()),
                'obstacles_starting_point_sampling_attempts': float(self._starting_point_sampling_attempts),
                'obstacles_initial_target_point_distance': self._get_initial_target_point_distance()
                }

        if self._use_target_points:
            if self._robot_scene.num_robots > 1:
                for i in range(self._robot_scene.num_robots):
                    info['obstacles_num_target_points_reached_r' + str(i)] = \
                        float(self.get_num_target_points_reached(robot=i))
                    info['obstacles_initial_target_point_distance_r' + str(i)] = \
                        self._get_initial_target_point_distance(robot=i)

            for i in range(self._robot_scene.num_robots):
                if len(self._target_point_sampling_attempts_list[i]) > 0:
                    mean_sampling_attempts = float(np.mean(self._target_point_sampling_attempts_list[i]))
                else:
                    mean_sampling_attempts = 0.0
                info['obstacles_target_point_sampling_attempts_r' + str(i)] = mean_sampling_attempts

            if self._log_target_point_relative_pos_min_max:
                info['obstacles_target_point_relative_pos_min_x'] = self._target_point_relative_pos_min_max_log[0][0]
                info['obstacles_target_point_relative_pos_min_y'] = self._target_point_relative_pos_min_max_log[0][1]
                info['obstacles_target_point_relative_pos_min_z'] = self._target_point_relative_pos_min_max_log[0][2]
                info['obstacles_target_point_relative_pos_max_x'] = self._target_point_relative_pos_min_max_log[1][0]
                info['obstacles_target_point_relative_pos_max_y'] = self._target_point_relative_pos_min_max_log[1][1]
                info['obstacles_target_point_relative_pos_max_z'] = self._target_point_relative_pos_min_max_log[1][2]

        if self._braking_timeout:
            info['obstacles_episodes_with_braking_timeout'] = 1.0
        else:
            info['obstacles_episodes_with_braking_timeout'] = 0.0

        if self._braking_duration_list:
            info['obstacles_braking_duration_mean'] = np.mean(self._braking_duration_list)
            info['obstacles_braking_duration_max'] = np.max(self._braking_duration_list)

        if self._active_braking_duration_list:
            info['obstacles_active_braking_duration_mean'] = np.mean(self._active_braking_duration_list)
            info['obstacles_active_braking_duration_max'] = np.max(self._active_braking_duration_list)
            if self._active_braking_influence_time_list:
                info['obstacles_active_braking_influence_time_mean'] = np.mean(self._active_braking_influence_time_list)
                info['obstacles_active_braking_influence_time_max'] = np.max(self._active_braking_influence_time_list)
        else:
            # episode without braking influence;
            # Note: To compute the mean of the active_braking_duration (execution time of activated braking
            # trajectories) and the braking_influence_time (actual influence time, equals active_braking_duration if
            # the robot is completely stopped), episodes with 0 values need to be neglected
            info['obstacles_active_braking_duration_mean'] = 0
            info['obstacles_active_braking_duration_max'] = 0
            info['obstacles_active_braking_influence_time_mean'] = 0
            info['obstacles_active_braking_influence_time_max'] = 0

        if info['obstacles_time_influenced_by_braking_trajectory'] == 0.0:
            info['obstacles_episodes_without_influence_by_braking_trajectory'] = 1.0
        else:
            info['obstacles_episodes_without_influence_by_braking_trajectory'] = 0.0
        if info['obstacles_time_influenced_by_braking_trajectory_collision'] == 0.0:
            info['obstacles_episodes_without_influence_by_braking_trajectory_collision'] = 1.0
        else:
            info['obstacles_episodes_without_influence_by_braking_trajectory_collision'] = 0.0
        if info['obstacles_time_influenced_by_braking_trajectory_torque'] == 0.0:
            info['obstacles_episodes_without_influence_by_braking_trajectory_torque'] = 1.0
        else:
            info['obstacles_episodes_without_influence_by_braking_trajectory_torque'] = 0.0

        if self._log_obstacle_data:
            for obstacle in self._obstacle_list:
                info['obstacles_link_data_' + obstacle.name] = [obstacle.link_data[i].export_metrics() for i in
                                                                range(len(obstacle.link_data))]

            for i in range(len(self._links)):
                export_link_pair = [self._links[i].closest_point_active or
                                    self._links[self._links[i].self_collision_links[j]].closest_point_active
                                    for j in range(len(self._links[i].self_collision_links))]
                info['obstacles_self_collision_data_link_' + str(i) + '_' + self._links[i].name] = \
                    self._links[i].self_collision_data.export_metrics(export_link_pair=export_link_pair)

        if self._print_stats:
            if (self._episode_counter % print_stats_every_n_episodes) == 0:
                logging.info("Sampling attempts starting point: " + str(self._starting_point_sampling_attempts))
                if self._braking_duration_list:
                    logging.info("Mean braking duration: " + str(info['obstacles_braking_duration_mean']))
                    logging.info("Max braking duration: " + str(info['obstacles_braking_duration_max']))
                if self._use_target_points:
                    logging.info("Number of target points reached: " +
                                 str(info['obstacles_num_target_points_reached']))
                    if self._robot_scene.num_robots > 1:
                        for i in range(self._robot_scene.num_robots):
                            logging.info("    - Robot " + str(i) + ": " + str(
                                info['obstacles_num_target_points_reached_r' + str(i)]))
                    logging.info("Number of attempts to find a valid target point")
                    for i in range(self._robot_scene.num_robots):
                        logging.info("    - Robot " + str(i) + ": " + str(
                            info['obstacles_target_point_sampling_attempts_r' + str(i)]))

        return info

    def _specify_observed_points(self, link_name, link_index):
        observed_points = []

        link_state = p.getLinkState(bodyUniqueId=self._robot_scene.robot_id, linkIndex=link_index,
                                    computeLinkVelocity=False,
                                    computeForwardKinematics=True)
        com_offset = link_state[2]

        safety_distance = self._observed_point_safety_distance

        if self._observed_link_point_scene == 1:
            if link_name in self._robot_scene.get_link_names_for_multiple_robots("iiwa_link_6"):
                observed_points.append(
                    LinkPointBase(name="P0", offset=[0, -0.0, 0.02], bounding_sphere_radius=0.12, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))

        if self._observed_link_point_scene == 2:

            if link_name in self._robot_scene.get_link_names_for_multiple_robots("iiwa_link_3"):
                observed_points.append(
                    LinkPointBase(name="P0", offset=[0, 0.01, 0.06], bounding_sphere_radius=0.1, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))

                observed_points.append(
                    LinkPointBase(name="P1", offset=[0.00, 0.03, 0.19], bounding_sphere_radius=0.1, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))

            if link_name in self._robot_scene.get_link_names_for_multiple_robots("iiwa_link_4"):
                observed_points.append(
                    LinkPointBase(name="P0", offset=[0, -0.04, 0.02], bounding_sphere_radius=0.1, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))

                observed_points.append(
                    LinkPointBase(name="P1", offset=[0, -0.015, 0.16], bounding_sphere_radius=0.105, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))

            if link_name in self._robot_scene.get_link_names_for_multiple_robots("iiwa_link_5"):
                observed_points.append(
                    LinkPointBase(name="P0", offset=[0, -0.015, -0.11], bounding_sphere_radius=0.105, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))

            if link_name in self._robot_scene.get_link_names_for_multiple_robots("iiwa_link_6"):
                observed_points.append(
                    LinkPointBase(name="P0", offset=[0, -0.0, 0.02], bounding_sphere_radius=0.12, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))
        if self._observed_link_point_scene == 3:

            if link_name in self._robot_scene.get_link_names_for_multiple_robots("iiwa_link_3"):
                observed_points.append(
                    LinkPointBase(name="P1", offset=[0.00, 0.03, 0.19], bounding_sphere_radius=0.1, active=True,
                                  visualize_bounding_sphere=self._visualize_bounding_spheres,
                                  default_bounding_sphere_color=OBSERVED_POINT_NO_INFLUENCE_COLOR,
                                  num_clients=self._robot_scene.num_clients,
                                  safety_distance=safety_distance,
                                  ))

        return observed_points

    def _get_robot_index_from_link_name(self, link_name):
        # returns the robot index extracted from the link name, e.g. 1 for iiwa_link_4_r1
        # returns -1 if no link index is found and if multiple robots are in use, 0 otherwise
        if self._robot_scene.num_robots > 1:
            if re.match('^.*_r[0-9]+$', link_name):
                # e.g. extract 1 from linkname_r1
                return int(link_name.rsplit('_', 1)[1][1:])
            else:
                return -1
        else:
            return 0

    def _specify_self_collision_links(self, link_name, link_name_list):
        self_collision_link_names = []

        if self._robot_scene.num_robots > 1:
            if self._robot_scene.robot_name == "iiwa7":
                collision_between_robots_link_names = ["iiwa_base_adapter", "iiwa_link_0", "iiwa_link_1", "iiwa_link_2",
                                                       "iiwa_link_3", "iiwa_link_4", "iiwa_link_5", "iiwa_link_6",
                                                       "iiwa_link_7"]
            elif self._robot_scene.robot_name.startswith("armar6"):
                collision_between_robots_link_names = ["arm_t12", "arm_t23", "arm_t34", "arm_t45", "arm_t56", "arm_t67",
                                                       "arm_t78", "arm_t8", "hand_fixed"]
            elif self._robot_scene.robot_name.startswith("armar4"):
                collision_between_robots_link_names = ["arm_sho1", "arm_sho2", "arm_sho3", "arm_sho4", "arm_elb1",
                                                       "arm_elb2", "arm_wri2"]
            else:
                collision_between_robots_link_names = []

            for i in range(self._robot_scene.num_robots - 1):
                if link_name in self._robot_scene.get_link_names_for_multiple_robots(
                        collision_between_robots_link_names,
                        robot_indices=[i]):
                    self_collision_link_names += (self._robot_scene.get_link_names_for_multiple_robots(
                        collision_between_robots_link_names, robot_indices=np.arange(i + 1,
                                                                                     self._robot_scene.num_robots)))

            if self._robot_scene.robot_name.startswith("armar6"):
                if link_name in self._robot_scene.get_link_names_for_multiple_robots(
                        collision_between_robots_link_names):
                    self_collision_link_names += ["platform", "torso", "lower_neck", "middle_neck", "upper_neck"]
                for i in range(self._robot_scene.num_robots):
                    if link_name in self._robot_scene.get_link_names_for_multiple_robots(["hand_fixed"],
                                                                                         robot_indices=[i]):
                        self_collision_link_names += self._robot_scene.get_link_names_for_multiple_robots(
                            ["arm_t12", "arm_t23"], robot_indices=[i])

            if self._robot_scene.robot_name.startswith("armar4"):
                if link_name in self._robot_scene.get_link_names_for_multiple_robots(
                        ["arm_sho3", "arm_sho4", "arm_elb1",
                         "arm_elb2", "arm_wri2"]):
                    self_collision_link_names += ["torso_pitch"]
                if link_name in self._robot_scene.get_link_names_for_multiple_robots(
                        ["arm_elb1", "arm_elb2", "arm_wri2"]):
                    self_collision_link_names += ["neck_2", "neck_3", "head_2"]
                if link_name in self._robot_scene.get_link_names_for_multiple_robots(
                        ["arm_elb2", "arm_wri2"]):
                    self_collision_link_names += ["torso_yaw", "root"]
                if link_name in self._robot_scene.get_link_names_for_multiple_robots(["arm_wri2"]):
                    self_collision_link_names += self._robot_scene.get_link_names_for_multiple_robots(
                        ["leg_hip1", "leg_hip2", "leg_hip3"])

        self_collision_link_indices = []
        for i in range(len(self_collision_link_names)):
            link_index = None
            for j in range(len(link_name_list)):
                if link_name_list[j] == self_collision_link_names[i]:
                    link_index = j
                    self_collision_link_indices.append(link_index)
                    break
            if link_index is None:
                raise ValueError(self_collision_link_names[i] + " is not a valid link name")

        return self_collision_link_indices

    def get_starting_point_joint_pos(self, minimum_initial_distance_to_obstacles=None,
                                     minimum_distance_self_collision=None):
        if minimum_initial_distance_to_obstacles is None:
            if self._robot_scene.robot_name == "armar6_x4":
                minimum_initial_distance_to_obstacles = self._closest_point_safety_distance + 0.04
            else:
                minimum_initial_distance_to_obstacles = self._closest_point_safety_distance + 0.09
        if minimum_distance_self_collision is None:
            if self._robot_scene.robot_name == "armar6_x4":
                minimum_distance_self_collision = self._closest_point_safety_distance + 0.015
            else:
                minimum_distance_self_collision = self._closest_point_safety_distance + 0.04

        starting_point_joint_pos, _, attempts_counter = self._get_collision_free_robot_position(
            minimum_initial_distance_to_obstacles=
            minimum_initial_distance_to_obstacles,
            minimum_distance_self_collision=
            minimum_distance_self_collision,
            cartesian_range=self._starting_point_cartesian_range,
            euler_angle_range=None,
            attempts=20000)

        self._starting_point_sampling_attempts = attempts_counter

        return starting_point_joint_pos

    def _add_target_point(self, robot=0, minimum_initial_distance_to_obstacles=None,
                          minimum_distance_self_collision=None):

        if minimum_initial_distance_to_obstacles is None:
            minimum_initial_distance_to_obstacles = self._closest_point_safety_distance + 0.09
        if minimum_distance_self_collision is None:
            minimum_distance_self_collision = self._closest_point_safety_distance + 0.00

        euler_angle_range = None
        attempts = 25000

        self._target_point_joint_pos_list[robot], target_point_pos, attempts_counter = \
            self._get_collision_free_robot_position(
                minimum_initial_distance_to_obstacles=minimum_initial_distance_to_obstacles,
                minimum_distance_self_collision=minimum_distance_self_collision,
                cartesian_range=self._target_point_cartesian_range,
                check_initial_torque=True,
                euler_angle_range=euler_angle_range,
                robot=robot,
                attempts=attempts)

        self._target_point_sampling_attempts_list[robot].append(attempts_counter)
        logging.debug("Target point position robot %s: %s", robot, target_point_pos)

        color_index = robot if self._target_point_sequence != TARGET_POINT_SINGLE else 0
        self._target_point_list[robot].append(self._add_obstacle(enable_collisions=False,
                                                                 create_collision_shape=False,
                                                                 pos=target_point_pos,
                                                                 shape=p.GEOM_SPHERE, radius=self._target_point_radius,
                                                                 observed_link_names=
                                                                 self._robot_scene.get_link_names_for_multiple_robots(
                                                                   self._target_link_name),
                                                                 name="Target-" + str(robot) + "-" + str(
                                                                   len(self._target_point_list)),
                                                                 is_static=True,
                                                                 color=self.get_target_point_color(color_index)))

    def get_target_point_color(self, robot=0, transparency=0.5):
        if robot == 0:
            target_point_color = (0, 1, 0, transparency)
        elif robot == 1:
            target_point_color = (126 / 255, 47 / 255, 142 / 255, transparency)
        elif robot == 2:
            target_point_color = (23 / 255, 190 / 255, 207 / 255, transparency)  # dark turquoise
        else:
            target_point_color = (255 / 255, 153 / 255, 0 / 255, transparency)  # orange

        return target_point_color

    def _get_collision_free_robot_position(self, minimum_initial_distance_to_obstacles, minimum_distance_self_collision,
                                           cartesian_range, euler_angle_range=None, check_initial_torque=True,
                                           robot=None, static_joint_pos=None, attempts=10000):

        valid_pos_found = False
        attempts_counter = 0
        if robot is None:  # all joints
            manip_joint_indices_robot = self._manip_joint_indices
        else:
            manip_joint_indices_robot = np.array(self._robot_scene.get_manip_joint_indices_per_robot(robot_index=robot))

        joint_limit_indices_robot = []
        static_joint_index_counter = -1
        static_joint_indices = []
        if static_joint_pos is None:
            use_static_joint_pos_from_argument = False
            static_joint_pos = []
        else:
            use_static_joint_pos_from_argument = True

        for i in range(len(self._manip_joint_indices)):
            if self._manip_joint_indices[i] not in manip_joint_indices_robot:
                static_joint_index_counter += 1
                static_joint_indices.append(self._manip_joint_indices[i])
                if not use_static_joint_pos_from_argument:
                    target_pos = self._target_position[i]
                    static_joint_pos.append(target_pos)
                    target_vel = 0
                else:
                    # use positions given by the argument static_joint_pos to generate a starting pos in multiple steps
                    target_pos = static_joint_pos[static_joint_index_counter]
                    target_vel = 0

                p.resetJointState(bodyUniqueId=self._robot_scene.robot_id,
                                  jointIndex=self._manip_joint_indices[i],
                                  targetValue=target_pos,
                                  targetVelocity=target_vel,
                                  physicsClientId=self._obstacle_client_id)

            else:
                joint_limit_indices_robot.append(i)

        joint_limit_indices_robot = np.array(joint_limit_indices_robot)

        if check_initial_torque and static_joint_pos:
            # set motor control for static joints
            self._robot_scene.set_motor_control(target_positions=static_joint_pos,
                                                physics_client_id=self._obstacle_client_id,
                                                manip_joint_indices=static_joint_indices)

        while not valid_pos_found:
            valid_pos_found = True
            self_collision_ignored_links = None
            reason = None
            attempts_counter += 1
            random_pos = np.random.uniform(
                np.array(self._robot_scene.joint_lower_limits_continuous)[joint_limit_indices_robot],
                np.array(self._robot_scene.joint_upper_limits_continuous)[joint_limit_indices_robot])

            # set position of affected links
            self.set_robot_position_in_obstacle_client(manip_joint_indices=manip_joint_indices_robot,
                                                       target_position=random_pos)

            if self._use_target_points:
                for i in range(self._robot_scene.num_robots):
                    if robot is None or robot == i:
                        target_link_pos, target_link_orn = self._target_link_point_list[i].get_position(actual=None,
                                                                                                        return_orn=True)
                        if target_link_pos[0] < cartesian_range[0][0] \
                                or target_link_pos[0] > cartesian_range[0][1] or \
                                target_link_pos[1] < cartesian_range[1][0] \
                                or target_link_pos[1] > cartesian_range[1][1] or \
                                target_link_pos[2] < cartesian_range[2][0] \
                                or target_link_pos[2] > cartesian_range[2][1]:
                            valid_pos_found = False
                            break

                        if euler_angle_range is not None:
                            # check orientation of the end effector
                            # euler_angle_range e.g. [[alpha_min, alpha_max], [beta_min, beta_max],
                            # [gamma_min, gamma_max]]
                            target_link_orn_euler = p.getEulerFromQuaternion(target_link_orn)
                            if target_link_orn_euler[0] < euler_angle_range[0][0] \
                                    or target_link_orn_euler[0] > euler_angle_range[0][1] or \
                                    target_link_orn_euler[1] < euler_angle_range[1][0] \
                                    or target_link_orn_euler[1] > euler_angle_range[1][1] or \
                                    target_link_orn_euler[2] < euler_angle_range[2][0] \
                                    or target_link_orn_euler[2] > euler_angle_range[2][1]:
                                valid_pos_found = False
            else:
                target_link_pos = None

            if valid_pos_found:
                for i in range(len(self._obstacle_list)):
                    for j in range(len(self._obstacle_list[i].observed_links)):
                        link_index = self._obstacle_list[i].observed_links[j]
                        if self._links[link_index].closest_point_active and \
                                (robot is None or self._links[link_index].robot_index == robot):
                            pos_obs, pos_rob, distance = self._compute_closest_points(
                                p.getClosestPoints(bodyA=self._obstacle_list[i].id,
                                                   bodyB=self._robot_scene.robot_id,
                                                   distance=minimum_initial_distance_to_obstacles + 0.005,
                                                   linkIndexA=self._obstacle_list[
                                                       i].last_link,
                                                   linkIndexB=link_index,
                                                   physicsClientId=self._obstacle_client_id))

                            if distance is not None and distance < minimum_initial_distance_to_obstacles:
                                valid_pos_found = False
                                reason = "Collision: LinkIndex: " + str(link_index) + ", ObstacleIndex: " + str(i)
                                break
                    if not valid_pos_found:
                        break

            if valid_pos_found:
                for i in range(len(self._links)):
                    for j in range(len(self._links[i].self_collision_links)):
                        if (self._links[i].closest_point_active or self._links[
                            self._links[i].self_collision_links[j]].closest_point_active) \
                                and (robot is None or (self._links[i].robot_index == robot or
                                                       self._links[self._links[i].self_collision_links[j]].robot_index
                                                       == robot)):
                            pos_rob_a, pos_rob_b, distance = self._compute_closest_points(
                                p.getClosestPoints(bodyA=self._robot_scene.robot_id,
                                                   bodyB=self._robot_scene.robot_id,
                                                   distance=minimum_distance_self_collision + 0.005,
                                                   linkIndexA=i,
                                                   linkIndexB=self._links[i].self_collision_links[j],
                                                   physicsClientId=self._obstacle_client_id))

                            if distance is not None and distance < minimum_distance_self_collision:
                                if robot is not None and attempts_counter > attempts * 0.75 and (
                                        (self._links[i].robot_index == robot and self._links[
                                            self._links[i].self_collision_links[j]].closest_point_active) or (
                                                self._links[self._links[i].self_collision_links[j]].robot_index
                                                == robot and self._links[i].closest_point_active)):
                                    self_collision_ignored_links = [i, self._links[i].self_collision_links[j]]
                                    # ignore self-collisions when finding target point positions if attempts_counter
                                    # is high and if the colliding link can be actively moved (closest_point_actice)
                                else:
                                    valid_pos_found = False
                                    reason = "Self-collision: " \
                                             "[" + str(i) + ", " + str(self._links[i].self_collision_links[j]) + "]"
                                    break
                    if not valid_pos_found:
                        break

            if valid_pos_found and check_initial_torque:
                self._robot_scene.set_motor_control(target_positions=random_pos,
                                                    physics_client_id=self._obstacle_client_id,
                                                    manip_joint_indices=manip_joint_indices_robot)
                p.stepSimulation(physicsClientId=self._obstacle_client_id)

                actual_joint_torques = self._robot_scene.get_actual_joint_torques(
                    physics_client_id=self._obstacle_client_id,
                    manip_joint_indices=manip_joint_indices_robot)

                normalized_joint_torques = normalize(actual_joint_torques,
                                                     self._torque_limits[:, joint_limit_indices_robot])

                if np.any(np.abs(normalized_joint_torques) > 1):
                    if robot is not None and attempts_counter > attempts / 2:
                        # ignore torque violation to find at least a collision-free position for a target point
                        logging.warning("Ignored torque violation to find a collision-free target point. Robot: %s, "
                                        "Normalized torques: %s", robot, normalized_joint_torques)
                    else:
                        valid_pos_found = False
                        reason = "Torque violation: " + str(normalized_joint_torques)

            if valid_pos_found and self_collision_ignored_links is not None:
                logging.warning("Ignored self_collision to find a collision-free target point. Robot: %s, "
                                "Ignored links: %s", robot, self_collision_ignored_links)

            if not valid_pos_found and attempts is not None and reason is not None and attempts_counter >= attempts:
                raise ValueError("Could not find a valid collision-free robot position. "
                                 + "Reason: " + reason
                                 + ", minimum_initial_distance_to_obstacles=" +
                                 str(minimum_initial_distance_to_obstacles)
                                 + ", minimum_distance_self_collision=" + str(minimum_distance_self_collision)
                                 + ", cartesian_range=" + str(cartesian_range)
                                 + ", check_initial_torque=" + str(check_initial_torque)
                                 + ", robot=" + str(robot)
                                 + ", attempts=" + str(attempts))

        return random_pos, target_link_pos, attempts_counter

    def get_target_point_observation(self, compute_relative_pos_norm=False, compute_target_point_joint_pos_norm=False):
        relative_pos_norm_chain = []
        target_point_active_observation = []
        target_point_joint_pos_norm_chain = []
        target_point_pos_norm_chain = []

        if self._use_target_points:
            for i in range(self._robot_scene.num_robots):
                self._target_point_reached_list[i] = False
                if self._sample_new_target_point_list[i]:
                    self._add_target_point(robot=i, minimum_distance_self_collision=self._closest_point_safety_distance)
                    self._sample_new_target_point_list[i] = False
                    self._target_point_active_list[i] = True
                    self._target_point_pos_norm_list[i] = None
                    self._target_point_joint_pos_norm_list[i] = None
                    if self._target_point_sequence == TARGET_POINT_SINGLE:
                        for j in range(self._robot_scene.num_robots):
                            self._initial_target_point_distance_list[j].append(np.nan)
                    else:
                        self._initial_target_point_distance_list[i].append(np.nan)

            for i in range(self._robot_scene.num_robots):
                if self._target_point_sequence == TARGET_POINT_SINGLE:
                    target_point_index = np.where(self._target_point_active_list)[0][0]
                    # return index of first true element -> active target point
                else:
                    target_point_index = i
                if self._target_point_active_list[i] or self._target_point_sequence == TARGET_POINT_SINGLE:
                    if self._target_point_sequence == TARGET_POINT_ALTERNATING:
                        target_point_active_observation.append(1.0)  # to indicate that the target point is active
                    target_point_pos = \
                        np.array(self._target_point_list[target_point_index][-1].get_position(actual=False))
                    self._last_target_point_distance_list[i] = np.linalg.norm(
                        target_point_pos - self._target_link_pos_list[i])
                    if np.isnan(self._initial_target_point_distance_list[i][-1]):
                        self._initial_target_point_distance_list[i][-1] = self._last_target_point_distance_list[i]

                    if self._target_point_active_list[i]:
                        if self._target_point_pos_norm_list[i] is None:
                            self._target_point_pos_norm_list[i] = \
                                list(normalize(target_point_pos, self._target_point_cartesian_range_min_max))
                        if compute_target_point_joint_pos_norm and self._target_point_joint_pos_norm_list[i] is None:
                            joint_indices = np.array(self._robot_scene.get_manip_joint_indices_per_robot(robot_index=i))
                            joint_limit_indices_robot = []
                            for k in range(len(self._manip_joint_indices)):
                                if self._manip_joint_indices[k] in joint_indices:
                                    joint_limit_indices_robot.append(k)
                            joint_limit_indices_robot = np.array(joint_limit_indices_robot)
                            joint_lower_limits = self._robot_scene.joint_lower_limits_continuous[
                                joint_limit_indices_robot]
                            joint_upper_limits = self._robot_scene.joint_upper_limits_continuous[
                                joint_limit_indices_robot]
                            self._target_point_joint_pos_norm_list[i] = [
                                -1 + 2 * (self._target_point_joint_pos_list[i][j] -
                                          joint_lower_limits[j]) /
                                (joint_upper_limits[j] - joint_lower_limits[j])
                                for j in range(len(self._target_point_joint_pos_list[i]))]

                    if compute_relative_pos_norm:
                        relative_pos = target_point_pos - self._target_link_pos_list[i]
                        relative_pos_norm = normalize(relative_pos, self._target_point_relative_pos_min_max)
                        relative_pos_norm_clip = [min(max(relative_pos_norm[i], -1), 1) for i in range(3)]
                        if self._log_target_point_relative_pos_min_max:
                            if not list(relative_pos_norm) == relative_pos_norm_clip:
                                logging.warning("Target point relative pos norm %s clipped", relative_pos_norm)
                            for j in range(3):
                                self._target_point_relative_pos_min_max_log[0][j] = \
                                    min(self._target_point_relative_pos_min_max_log[0][j], relative_pos[j])
                                self._target_point_relative_pos_min_max_log[1][j] = \
                                    max(self._target_point_relative_pos_min_max_log[1][j], relative_pos[j])
                        relative_pos_norm_chain.extend(relative_pos_norm_clip)
                else:
                    if self._target_point_sequence == TARGET_POINT_ALTERNATING:
                        target_point_active_observation.append(0.0)
                    self._target_point_pos_norm_list[i] = [0, 0, 0]
                    if compute_relative_pos_norm:
                        relative_pos_norm_chain.extend([0, 0, 0])
                    if compute_target_point_joint_pos_norm:
                        self._target_point_joint_pos_norm_list[i] = [0.0] * len(self._target_point_joint_pos_list[i])

            if self._target_point_sequence == TARGET_POINT_SINGLE:
                target_point_pos_norm_chain = self._target_point_pos_norm_list[target_point_index]
            else:
                target_point_pos_norm_chain = list(
                    itertools.chain.from_iterable(self._target_point_pos_norm_list))
                # merge all list entries to a single list
            if compute_target_point_joint_pos_norm:
                if self._target_point_sequence == TARGET_POINT_SINGLE:
                    target_point_joint_pos_norm_chain = list(self._target_point_joint_pos_norm_list[target_point_index])
                else:
                    target_point_joint_pos_norm_chain = list(itertools.chain.from_iterable(
                        self._target_point_joint_pos_norm_list))
            else:
                target_point_joint_pos_norm_chain = []

        return target_point_pos_norm_chain, relative_pos_norm_chain, target_point_joint_pos_norm_chain, \
               target_point_active_observation

    def get_target_point_reward(self, normalize_distance_reward_to_initial_target_point_distance=False):
        reward = 0
        if self._use_target_points:
            for i in range(self._robot_scene.num_robots):
                if (self._target_point_active_list[i] or self._target_point_reached_list[i]) and \
                        normalize_distance_reward_to_initial_target_point_distance:
                    if self._target_point_sequence == TARGET_POINT_SINGLE:
                        initial_distance = []
                        for j in range(self._robot_scene.num_robots):
                            initial_distance.append(self._initial_target_point_distance_list[j][-1])
                        distance_normalization = min(initial_distance)
                    else:
                        distance_normalization = self._initial_target_point_distance_list[i][-1]
                    if distance_normalization == 0:
                        distance_normalization += 0.0000001
                else:
                    distance_normalization = 1

                if self._target_point_active_list[i]:
                    if self._target_point_sequence == TARGET_POINT_SINGLE:
                        current_distance_list = []
                        for j in range(self._robot_scene.num_robots):
                            current_distance_list.append(np.linalg.norm(self._target_point_pos_list[i] -
                                                                        self._target_link_pos_list[j]))
                        current_distance = min(current_distance_list)
                        last_distance = min(self._last_target_point_distance_list)
                    else:
                        current_distance = np.linalg.norm(self._target_point_pos_list[i] -
                                                          self._target_link_pos_list[i])
                        last_distance = self._last_target_point_distance_list[i]

                    reward = reward + (last_distance - current_distance) / (self._trajectory_time_step
                                                                            * distance_normalization)
                elif self._target_point_reached_list[i]:
                    if self._target_point_sequence == TARGET_POINT_SINGLE:
                        last_distance = min(self._last_target_point_distance_list)
                    else:
                        last_distance = self._last_target_point_distance_list[i]
                    reward = reward + last_distance / (self._trajectory_time_step * distance_normalization)
                    reward += self._target_point_reached_reward_bonus

        return reward

    def get_braking_trajectory_punishment(self, minimum_distance_max_threshold, maximum_torque_min_threshold):
        # computes a punishment factor within [0, 1] based on the minimum distance and the maximum torque
        # occurring during the braking trajectory
        minimum_distance_punishment = 0
        maximum_torque_punishment = 0

        if self._use_braking_trajectory_method:
            if self._braking_trajectory_minimum_distance < self._closest_point_safety_distance:
                minimum_distance_punishment = 1.0
            else:
                if self._braking_trajectory_minimum_distance < minimum_distance_max_threshold:
                    minimum_distance_punishment = max(
                        min((minimum_distance_max_threshold - self._braking_trajectory_minimum_distance) /
                            (minimum_distance_max_threshold - self._closest_point_safety_distance), 1), 0) ** 2

                if self._check_braking_trajectory_torque_limits:
                    maximum_torque_punishment = max(
                        min((self._braking_trajectory_maximum_rel_torque
                             - maximum_torque_min_threshold) /
                            (1 - maximum_torque_min_threshold), 1), 0) ** 2

        return minimum_distance_punishment, maximum_torque_punishment

    def _add_obstacles(self):
        if self._robot_scene.robot_name == "iiwa7":

            observed_link_names = ["iiwa_link_1", "iiwa_link_2", "iiwa_link_3", "iiwa_link_4", "iiwa_link_5",
                                   "iiwa_link_6", "iiwa_link_7"]

            if self._obstacle_scene == 1:
                table_color = (0.02, 0.02, 0.4, 0.5)
                plane_points = [[-0.6, -0.8, 0], [0.6, -0.8, 0], [-0.6, 0.8, 0]]
                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions, pos=None,
                                       shape=p.GEOM_PLANE, plane_points=plane_points,
                                       observed_link_names=self._robot_scene.get_link_names_for_multiple_robots(
                                           observed_link_names), name="Table", plane_collision_shape_factor=2.0,
                                       is_static=True, color=table_color))

            if 2 <= self._obstacle_scene <= 4:
                # table
                table_color = (0.02, 0.02, 0.4, 0.5)
                plane_points = [[-0.6, -0.8, 0], [0.6, -0.8, 0], [-0.6, 0.8, 0]]
                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions, pos=None,
                                       shape=p.GEOM_PLANE, plane_points=plane_points,
                                       observed_link_names=self._robot_scene.get_link_names_for_multiple_robots(
                                           observed_link_names), name="Table", plane_collision_shape_factor=1.0,
                                       is_static=True, color=table_color))

                # virtual walls
                wall_color = (0.8, 0.8, 0.8, 0.1)
                wall_height = 1.2  # meter
                # left wall
                plane_points = [[-0.6, -0.8, 0], [0.6, -0.8, 0], [-0.6, -0.8, wall_height]]
                plane_orn = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions, pos=None,
                                       shape=p.GEOM_PLANE, plane_points=plane_points, orn=plane_orn,
                                       observed_link_names=self._robot_scene.get_link_names_for_multiple_robots(
                                           observed_link_names), name="Wall left", plane_collision_shape_factor=1.0,
                                       is_static=True, color=wall_color))
                # front wall
                plane_points = [[0.6, -0.8, 0], [0.6, -0.8, wall_height], [0.6, 0.8, 0]]
                plane_orn = p.getQuaternionFromEuler([0, np.pi / 2, 0])
                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions,
                                       pos=None, shape=p.GEOM_PLANE, plane_points=plane_points, orn=plane_orn,
                                       observed_link_names=self._robot_scene.get_link_names_for_multiple_robots(
                                           observed_link_names), name="Wall front", plane_collision_shape_factor=1.0,
                                       is_static=True, color=wall_color))
                # right wall
                plane_points = [[-0.6, 0.8, 0], [0.6, 0.8, 0.0], [-0.6, 0.8, wall_height]]
                plane_orn = p.getQuaternionFromEuler([np.pi / 2, 0, 0])
                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions, pos=None,
                                       shape=p.GEOM_PLANE, plane_points=plane_points, orn=plane_orn,
                                       observed_link_names=self._robot_scene.get_link_names_for_multiple_robots(
                                           observed_link_names), name="Wall right", plane_collision_shape_factor=1.0,
                                       is_static=True, color=wall_color))

                # back wall
                plane_points = [[-0.6, -0.8, 0], [-0.6, -0.8, wall_height], [-0.6, 0.8, 0]]
                plane_orn = p.getQuaternionFromEuler([0, np.pi / 2, 0])
                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions, pos=None,
                                       shape=p.GEOM_PLANE, plane_points=plane_points, orn=plane_orn,
                                       observed_link_names=self._robot_scene.get_link_names_for_multiple_robots(
                                           observed_link_names), name="Wall back", plane_collision_shape_factor=1.0,
                                       is_static=True, color=wall_color))

                if self._obstacle_scene >= 3:
                    # add monitor
                    if self._obstacle_scene == 3:
                        monitor_file_name = "monitor_no_pivot"
                        obstacle_name = "Monitor"
                    else:
                        monitor_file_name = "monitor_pivot"
                        obstacle_name = "Monitor (rotated)"

                    self._obstacle_list.append(
                        self._add_obstacle(enable_collisions=self._activate_obstacle_collisions, pos=[0.23, 0.045, 0.0],
                                           urdf_file_name=monitor_file_name,
                                           observed_link_names=self._robot_scene.get_link_names_for_multiple_robots(
                                               observed_link_names), name=obstacle_name, is_static=True,
                                           color=(0.02, 0.02, 0.02, 0.65)))

            if self._obstacle_scene >= 5:
                logging.warning("Obstacle scene %s not defined for %s", self._obstacle_scene,
                                self._robot_scene.robot_name)

        if self._robot_scene.robot_name.startswith("armar6"):
            if self._obstacle_scene == 5:
                # define the floor as an obstacle to avoid collisions with the hands of armar 6:
                floor_color = (1, 1, 1, 0)
                z_coord = -0.0025   # slightly visible with 0.001
                plane_points = [[-1.0, -1.2, z_coord], [1.0, -1.2, z_coord], [-1.0, 1.2, z_coord]]
                self._obstacle_list.append(
                    self._add_obstacle(enable_collisions=self._activate_obstacle_collisions, pos=None,
                                       shape=p.GEOM_PLANE, plane_points=plane_points,
                                       observed_link_names=self._robot_scene.get_link_names_for_multiple_robots(
                                           ["hand_fixed"]), name="Floor", plane_collision_shape_factor=1.0,
                                       is_static=True, color=floor_color))

            if 1 <= self._obstacle_scene <= 4 or self._obstacle_scene >= 6:
                logging.warning("Obstacle scene %s not defined for %s", self._obstacle_scene,
                                self._robot_scene.robot_name)

        self._update_links_in_use()

    def _add_obstacle(self, enable_collisions=False, create_collision_shape=True,
                      observed_link_names=[], *vargs, **kwargs):
        observed_links = []
        num_observed_points_per_link = []
        for i in range(len(self._links)):
            if self._links[i].name in observed_link_names:
                observed_links.append(i)
                num_observed_points_per_link.append(self._links[i].num_observed_points)
        obstacle = ObstacleSim(observed_links=observed_links, num_observed_points_per_link=num_observed_points_per_link,
                               simulation_client_id=self._simulation_client_id,
                               obstacle_client_id=self._obstacle_client_id,
                               use_real_robot=self._use_real_robot,
                               create_collision_shape=create_collision_shape,
                               num_clients=self._robot_scene.num_clients, *vargs, **kwargs)
        if not enable_collisions and create_collision_shape:
            self._deactivate_collision_detection(obstacle.id)
        return obstacle

    def _update_links_in_use(self):
        self._links_in_use = []
        for i in range(len(self._obstacle_list)):
            for j in range(len(self._obstacle_list[i].observed_links)):
                if self._obstacle_list[i].observed_links[j] not in self._links_in_use:
                    self._links_in_use.append(self._obstacle_list[i].observed_links[j])
                    if not self._links[self._obstacle_list[i].observed_links[j]].observed_points and not \
                            self._links[self._obstacle_list[i].observed_links[j]].observe_closest_point:
                        raise ValueError("No points to observe for link " +
                                         self._links[self._obstacle_list[i].observed_links[j]].name)

        self._links_in_use.sort()

    def _delete_all_target_points(self):
        for targetPoint in list(itertools.chain.from_iterable(self._target_point_list)):
            for j in range(self._robot_scene.num_clients):
                p.removeBody(targetPoint.id, physicsClientId=j)

        self._target_point_list = [[] for _ in range(self._robot_scene.num_robots)]

    def _deactivate_collision_detection(self, obstacle_id):
        for j in range(self._robot_scene.num_clients):
            for i in range(p.getNumJoints(self._robot_scene.robot_id)):
                p.setCollisionFilterPair(self._robot_scene.robot_id, obstacle_id, i,
                                         p.getNumJoints(obstacle_id) - 1, enableCollision=0, physicsClientId=j)

    def update(self, target_position, target_velocity, target_acceleration,
               actual_position, actual_velocity):

        pos_obs_debug = []
        pos_rob_debug = []

        self._target_position = target_position
        self._target_velocity = target_velocity

        self._last_actual_position = self._actual_position
        self._last_actual_velocity = self._actual_velocity
        self._actual_position = actual_position
        self._actual_velocity = actual_velocity

        for i in range(len(self._links)):
            self._links[i].clear_previous_timestep()

        self._obstacle_client_status = self.OBSTACLE_CLIENT_AT_OTHER_POSITION

        for obstacle in self._obstacle_list:
            obstacle.update()
            obstacle.clear_previous_timestep()

        if self._log_obstacle_data:
            # first step: actual values
            self.set_robot_position_in_obstacle_client(set_to_actual_values=True)
            obstacle_counter = - 1
            for obstacle in self._obstacle_list:
                obstacle_counter = obstacle_counter + 1
                for j in range(len(obstacle.observed_links)):
                    link_index = obstacle.observed_links[j]
                    # compute actual distance to closest point if enabled
                    if self._links[link_index].observe_closest_point:

                        pos_obs, pos_rob, distance = self._compute_closest_points(
                            p.getClosestPoints(bodyA=obstacle.id, bodyB=self._robot_scene.robot_id,
                                               distance=10,
                                               linkIndexA=obstacle.last_link,
                                               linkIndexB=link_index,
                                               physicsClientId=self._obstacle_client_id))
                        if obstacle_counter == self._debug_line_obstacle and j == self._debug_line_link and \
                                self._debug_line_point == 0:
                            pos_obs_debug = pos_obs
                            pos_rob_debug = pos_rob

                        obstacle.link_data[j].closest_point_distance_actual.append(distance)

                        if len(self._links[link_index].observed_points) > 0:
                            for k in range(len(self._links[link_index].observed_points)):
                                pos_rob = self._links[link_index].observed_points[k].get_position(actual=True)
                                pos_obs = obstacle.get_position(actual=True, pos_rob=pos_rob)
                                obstacle.link_data[j].observed_point_distance_actual[k].append(
                                    self._compute_distance(pos_obs, pos_rob,
                                                           radius_a=obstacle.bounding_sphere_radius,
                                                           radius_b=self._links[link_index].observed_points[
                                                               k].bounding_sphere_radius))
                                debug_line_point = k + 1 if self._links[link_index].observe_closest_point else k
                                if obstacle_counter == self._debug_line_obstacle and j == self._debug_line_link and \
                                        self._debug_line_point == debug_line_point:
                                    pos_obs_debug, pos_rob_debug = \
                                        self._consider_bounding_sphere(pos_obs, pos_rob,
                                                                       radius_a=obstacle.bounding_sphere_radius,
                                                                       radius_b=self._links[link_index].observed_points[
                                                                           k].bounding_sphere_radius)

            # self-collision
            for i in range(len(self._links)):
                for j in range(len(self._links[i].self_collision_links)):
                    # distance actual values
                    pos_rob_a, pos_rob_b, distance = self._compute_closest_points(
                        p.getClosestPoints(bodyA=self._robot_scene.robot_id,
                                           bodyB=self._robot_scene.robot_id,
                                           distance=10,
                                           linkIndexA=i,
                                           linkIndexB=self._links[i].self_collision_links[j],
                                           physicsClientId=self._obstacle_client_id))

                    self._links[i].self_collision_data.closest_point_distance_actual[j].append(distance)

            # second step: set points
            self.set_robot_position_in_obstacle_client(set_to_setpoints=True)
            for obstacle in self._obstacle_list:

                for j in range(len(obstacle.observed_links)):
                    link_index = obstacle.observed_links[j]

                    if self._links[link_index].observe_closest_point:
                        pos_obs, pos_rob, distance = \
                            self._compute_closest_points(p.getClosestPoints(bodyA=obstacle.id,
                                                                            bodyB=self._robot_scene.robot_id,
                                                                            distance=10,
                                                                            linkIndexA=obstacle.last_link,
                                                                            linkIndexB=link_index,
                                                                            physicsClientId=self._obstacle_client_id))
                        obstacle.link_data[j].closest_point_distance_set.append(distance)

                    if len(self._links[link_index].observed_points) > 0:
                        for k in range(len(self._links[link_index].observed_points)):
                            pos_rob = self._links[link_index].observed_points[k].get_position(actual=False)
                            pos_obs = obstacle.get_position(actual=False, pos_rob=pos_rob)

                            obstacle.link_data[j].observed_point_distance_set[k].append(
                                self._compute_distance(pos_obs, pos_rob,
                                                       radius_a=obstacle.bounding_sphere_radius,
                                                       radius_b=self._links[link_index].observed_points[
                                                           k].bounding_sphere_radius))

            # self-collision
            for i in range(len(self._links)):
                for j in range(len(self._links[i].self_collision_links)):
                    pos_rob_a, pos_rob_b, distance = self._compute_closest_points(
                        p.getClosestPoints(bodyA=self._robot_scene.robot_id,
                                           bodyB=self._robot_scene.robot_id,
                                           distance=10,
                                           linkIndexA=i,
                                           linkIndexB=self._links[i].self_collision_links[j],
                                           physicsClientId=self._obstacle_client_id))

                    self._links[i].self_collision_data.closest_point_distance_set[j].append(distance)

        if self._log_obstacle_data:
            if list(pos_obs_debug):
                line_color = [1, 0, 0]
                line_width = 2
                if self._debug_line is not None:
                    self._debug_line = p.addUserDebugLine(pos_obs_debug, pos_rob_debug, lineColorRGB=line_color,
                                                          lineWidth=line_width,
                                                          replaceItemUniqueId=self._debug_line,
                                                          physicsClientId=self._simulation_client_id)
                else:
                    self._debug_line = p.addUserDebugLine(pos_obs_debug, pos_rob_debug, lineColorRGB=line_color,
                                                          lineWidth=line_width,
                                                          physicsClientId=self._simulation_client_id)
            else:
                if self._debug_line is not None:
                    p.removeUserDebugItem(self._debug_line, physicsClientId=self._simulation_client_id)
                    self._debug_line = None

        if self._visualize_bounding_spheres:
            for j in range(len(self._links)):
                sphere_color = [0] * len(self._links[j].observed_points)
                for i in range(len(self._obstacle_list)):
                    for m in range(len(self._obstacle_list[i].observed_links)):
                        link_index = self._obstacle_list[i].observed_links[m]
                        if link_index == j:
                            for k in range(len(self._links[j].observed_points)):
                                if self._visualize_bounding_sphere_actual:
                                    distance = self._obstacle_list[i].link_data[m].observed_point_distance_actual[k][-1]
                                else:
                                    distance = self._obstacle_list[i].link_data[m].observed_point_distance_set[k][-1]

                                if distance < self._links[link_index].observed_points[k].safety_distance:
                                    sphere_color[k] = 2

                for k in range(len(self._links[j].observed_points)):
                    if sphere_color[k] == 2:
                        rgba_color = OBSERVED_POINT_VIOLATION_COLOR
                    else:
                        rgba_color = None

                    self._links[j].observed_points[k].update_bounding_sphere_position(
                        actual=self._visualize_bounding_sphere_actual)

                    if rgba_color is not None:
                        self._links[j].observed_points[k].set_bounding_sphere_color(rgba_color=rgba_color)

            if self._use_target_points:
                for k in range(len(self._target_link_point_list)):
                    self._target_link_point_list[k].update_bounding_sphere_position(
                        actual=self._visualize_bounding_sphere_actual)

        if self._use_target_points:
            # check if the target link point is close to the target point
            for i in range(self._robot_scene.num_robots):
                self._target_link_pos_list[i] = np.array(self._target_link_point_list[i].get_position(
                    actual=self._target_point_use_actual_position))

            for i in range(self._robot_scene.num_robots):
                if self._target_point_active_list[i]:
                    self._target_point_pos_list[i] = np.array(self._target_point_list[i][-1].get_position(actual=False))
                    if self._target_point_sequence == TARGET_POINT_SINGLE:
                        distance_list = []
                        for k in range(self._robot_scene.num_robots):
                            distance_list.append(self._compute_distance(self._target_link_pos_list[k],
                                                                        self._target_point_pos_list[i]))
                        distance = min(distance_list)
                        closest_robot = np.argmin(distance_list)
                    else:
                        distance = self._compute_distance(self._target_link_pos_list[i], self._target_point_pos_list[i])
                        closest_robot = i
                    if distance < self._target_point_list[i][-1].bounding_sphere_radius:
                        self._target_point_reached_list[i] = True
                        self._target_point_list[i][-1].make_invisible()
                        self._target_point_active_list[i] = False
                        self._num_target_points_reached_list[closest_robot] = \
                            self._num_target_points_reached_list[closest_robot] + 1
                        if self._target_point_sequence == TARGET_POINT_SIMULTANEOUS:
                            self._sample_new_target_point_list[i] = True
                        if self._target_point_sequence == TARGET_POINT_ALTERNATING:
                            target_point_active_robot_index = (i + 1) % self._robot_scene.num_robots
                            self._sample_new_target_point_list[target_point_active_robot_index] = True
                        if self._target_point_sequence == TARGET_POINT_SINGLE:
                            target_point_active_robot_index = np.random.randint(0, self._robot_scene.num_robots)
                            self._sample_new_target_point_list[target_point_active_robot_index] = True

    def _compute_closest_points(self, list_of_closest_points):
        pos_a = [0, 0, 0]
        pos_b = [0, 0, 0]
        closest_index = 0
        closest_distance = None
        if len(list_of_closest_points) > 0:
            closest_distance = list_of_closest_points[0][8]

            for i in range(1, len(list_of_closest_points)):
                if list_of_closest_points[i][8] < closest_distance:
                    closest_distance = list_of_closest_points[i][8]
                    closest_index = i

            pos_a = list_of_closest_points[closest_index][5]
            pos_b = list_of_closest_points[closest_index][6]

        return pos_a, pos_b, closest_distance

    def _compute_distance(self, pos_a, pos_b, radius_a=0, radius_b=0):
        return compute_distance_c(pos_a[0], pos_a[1], pos_a[2], pos_b[0], pos_b[1], pos_b[2], radius_a, radius_b)

    def set_robot_position_in_obstacle_client(self, manip_joint_indices=None, target_position=None,
                                              target_velocity=None, set_to_setpoints=False, set_to_actual_values=False,
                                              capture_frame=False):
        # set robot with physicsClientId self._obstacle_client_id to a specified position
        if set_to_setpoints and set_to_actual_values:
            raise ValueError("set_to_setpoints and set_to_actual_values are not allowed to be True at the same time")

        if set_to_setpoints:
            if self._obstacle_client_status == self.OBSTACLE_CLIENT_AT_TARGET_POSITION:
                return
            target_position = self._target_position
            target_velocity = self._target_velocity
            self._obstacle_client_status = self.OBSTACLE_CLIENT_AT_TARGET_POSITION

        if set_to_actual_values:
            if self._obstacle_client_status == self.OBSTACLE_CLIENT_AT_ACTUAL_POSITION:
                return
            target_position = self._actual_position
            target_velocity = self._actual_velocity
            self._obstacle_client_status = self.OBSTACLE_CLIENT_AT_ACTUAL_POSITION

        if not set_to_setpoints and not set_to_actual_values:
            self._obstacle_client_status = self.OBSTACLE_CLIENT_AT_OTHER_POSITION
            if target_velocity is None:
                target_velocity = [0.0] * len(target_position)

        for i in range(len(self._links)):
            self._links[i].clear_other_position_and_orn()

        if manip_joint_indices is None:
            manip_joint_indices = self._manip_joint_indices

        p.resetJointStatesMultiDof(bodyUniqueId=self._robot_scene.robot_id,
                                   jointIndices=manip_joint_indices,
                                   targetValues=[[pos] for pos in target_position],
                                   targetVelocities=[[vel] for vel in target_velocity],
                                   physicsClientId=self._obstacle_client_id)

        if capture_frame and self._robot_scene.capture_frame_function is not None:
            self._robot_scene.capture_frame_function()

    def _consider_bounding_sphere(self, pos_a, pos_b, radius_a, radius_b):
        if not np.array_equal(pos_a, pos_b):
            pos_diff = np.array(pos_b) - np.array(pos_a)
            pos_diff_norm = np.linalg.norm(pos_diff)
            pos_a_sphere = np.array(pos_a) + (radius_a / pos_diff_norm) * pos_diff
            pos_b_sphere = np.array(pos_a) + (1 - (radius_b / pos_diff_norm)) * pos_diff
            return pos_a_sphere, pos_b_sphere
        else:
            return [], []

    def get_braking_acceleration(self, last=False, no_shift=False):
        if last:
            self._valid_braking_trajectories['current'] = self._valid_braking_trajectories['last']
            self._valid_braking_trajectories['last'] = None
        min_distance = None
        max_torque = None
        if self._valid_braking_trajectories['current'] is not None:
            braking_acceleration = self._valid_braking_trajectories['current']['acceleration'][0]
            robot_stopped = False
            if self._valid_braking_trajectories['current']['min_distance']:
                min_distance = self._valid_braking_trajectories['current']['min_distance'][0]
            if self._valid_braking_trajectories['current']['max_torque']:
                max_torque = self._valid_braking_trajectories['current']['max_torque'][0]

            if not no_shift:
                if len(self._valid_braking_trajectories['current']['acceleration']) > 1:
                    for key, value in self._valid_braking_trajectories['current'].items():
                        self._valid_braking_trajectories['current'][key] = value[1:]
                else:
                    self._valid_braking_trajectories['current'] = None
        else:
            braking_acceleration = np.zeros(self._robot_scene.num_manip_joints)
            robot_stopped = True

        return braking_acceleration, robot_stopped, min_distance, max_torque

    def adapt_action(self, current_acc, current_vel, current_pos, target_acc,
                     time_step_counter=0):

        execute_braking_trajectory = False
        braking_trajectory_collision_free = True
        braking_trajectory_torque_limited = True
        adapted_acc = target_acc
        adaptation_punishment = 0  # adaptation punishment: 0: no adaptation, 1: braking trajectory executed
        min_distance = None  # after executing the adapted action
        max_torque = None
        # min_distance and max_torque after executing the adapted action (for reward calculation),
        # Both are None if execute_braking_trajectory == True
        # in addition, min_distance is None if the braking trajectory is not checked for collisions and
        # max_torque is None if the braking trajectory is not checked for torque limit violations

        if self._use_braking_trajectory_method:
            self._braking_trajectory = {'acceleration': [current_acc, target_acc],
                                        'velocity': [current_vel],
                                        'position': [current_pos],
                                        'min_distance': [],
                                        'max_torque': []}

            affected_observed_point = None

            if self._check_braking_trajectory_collisions:
                braking_trajectory_collision_free, affected_link_index_list, affected_observed_point \
                    = self._check_if_braking_trajectory_is_collision_free(time_step_counter)

            if self._check_braking_trajectory_torque_limits and braking_trajectory_collision_free:
                braking_trajectory_torque_limited, affected_link_index_list = \
                    self._check_if_braking_trajectory_is_torque_limited(time_step_counter)

            if not braking_trajectory_collision_free or not braking_trajectory_torque_limited:
                execute_braking_trajectory = True
                adaptation_punishment = 1.0

            self._braking_duration_list.append(self._braking_duration)

            if execute_braking_trajectory:
                if self._active_braking_influence_time == 0:
                    if len(self._braking_duration_list) > 0:
                        self._active_braking_duration_list.append(self._braking_duration_list[-1])
                    else:
                        self._active_braking_duration_list.append(0)  # start braking from the initial position
                self._active_braking_influence_time += self._trajectory_time_step

            else:
                if self._active_braking_influence_time != 0:
                    # the last action was adapted by the braking trajectory method
                    self._active_braking_influence_time_list.append(self._active_braking_influence_time)
                self._active_braking_influence_time = 0

                self._valid_braking_trajectories['last'] = self._valid_braking_trajectories['current']
                self._valid_braking_trajectories['current'] = self._braking_trajectory

                if self._valid_braking_trajectories['current']['min_distance']:
                    min_distance = self._valid_braking_trajectories['current']['min_distance'][0]
                    self._valid_braking_trajectories['current']['min_distance'] = \
                        self._valid_braking_trajectories['current']['min_distance'][1:]
                else:
                    min_distance = None
                if self._valid_braking_trajectories['current']['max_torque']:
                    max_torque = self._valid_braking_trajectories['current']['max_torque'][0]
                    self._valid_braking_trajectories['current']['max_torque'] = \
                        self._valid_braking_trajectories['current']['max_torque'][1:]
                else:
                    max_torque = None

                if self._braking_trajectory_length > 0:
                    for key in ['position', 'velocity', 'acceleration']:
                        self._valid_braking_trajectories['current'][key] = \
                            self._valid_braking_trajectories['current'][key][2:]
                        # remove current and next kinematic state
                else:
                    self._valid_braking_trajectories['current'] = None

            if self._visual_mode:
                # set colors for each link
                for i in range(len(self._links)):
                    if execute_braking_trajectory and i in affected_link_index_list:
                        if braking_trajectory_torque_limited:
                            if len(affected_link_index_list) == 1:
                                color = LINK_OBJECT_COLLISION_INFLUENCE_COLOR
                            else:
                                color = LINK_SELF_COLLISION_INFLUENCE_COLOR  # self collision
                        else:
                            color = LINK_TORQUE_INFLUENCE_COLOR  # potential torque violations
                        self._links[i].set_color(color)
                    else:
                        self._links[i].set_color(rgba_color=None)  # set color to default
                    if self._visualize_bounding_spheres and self._check_braking_trajectory_observed_points:
                        for k in range(len(self._links[i].observed_points)):
                            if affected_observed_point is not None and affected_observed_point[0] == i \
                                    and affected_observed_point[1] == k:
                                self._links[i].observed_points[k].set_bounding_sphere_color(
                                    rgba_color=OBSERVED_POINT_INFLUENCE_COLOR)
                            else:
                                self._links[i].observed_points[k].set_bounding_sphere_color(rgba_color=None)

        if braking_trajectory_collision_free:
            self._time_influenced_by_braking_trajectory_collision_list.append(0.0)
        else:
            self._time_influenced_by_braking_trajectory_collision_list.append(1.0)

        if braking_trajectory_torque_limited:
            self._time_influenced_by_braking_trajectory_torque_list.append(0.0)
        else:
            self._time_influenced_by_braking_trajectory_torque_list.append(1.0)

        return adapted_acc, execute_braking_trajectory, adaptation_punishment, min_distance, max_torque

    def _check_if_braking_trajectory_is_collision_free(self, time_step_counter=0):
        # execute the following step with the target acceleration and compute a braking trajectory after that step.
        # check for each time step if a collision occurred
        self._braking_trajectory_minimum_distance = np.inf
        self._braking_trajectory_maximum_rel_torque = 0

        robot_stopped = False
        collision_found = False
        affected_link_index_list = []
        affected_observed_point = None  # if affected: [link_index, point_index]
        minimum_distance = None

        time_since_start = np.linspace(self._trajectory_time_step / self._collision_checks_per_time_step,
                                       self._trajectory_time_step,
                                       self._collision_checks_per_time_step)
        while not robot_stopped and not collision_found:
            start_acceleration = self._braking_trajectory['acceleration'][-2]
            end_acceleration = self._braking_trajectory['acceleration'][-1]
            start_velocity = self._braking_trajectory['velocity'][-1]
            start_position = self._braking_trajectory['position'][-1]

            interpolated_position_batch = interpolate_position_batch(start_acceleration,
                                                                     end_acceleration,
                                                                     start_velocity,
                                                                     start_position, time_since_start,
                                                                     self._trajectory_time_step)

            for m in range(self._collision_checks_per_time_step):
                minimum_distance, collision_found, affected_link_index_list, affected_observed_point = \
                    self.get_minimum_distance(manip_joint_indices=self._manip_joint_indices,
                                              target_position=interpolated_position_batch[m])

                if minimum_distance < self._braking_trajectory_minimum_distance:
                    self._braking_trajectory_minimum_distance = minimum_distance

                if collision_found:
                    break

            if not collision_found:
                self._braking_trajectory['min_distance'].append(minimum_distance)
                # minimum distance at m == self._collision_checks_per_time_step - 1:

                # compute the target acceleration for the next decision step
                robot_stopped, braking_timeout = \
                    self._compute_next_braking_trajectory_time_step(start_position=interpolated_position_batch[-1])

                if braking_timeout:
                    collision_found = True

        if robot_stopped and not collision_found:
            return True, [], None
        else:
            return False, affected_link_index_list, affected_observed_point

    def get_minimum_distance(self, manip_joint_indices, target_position):
        # returns minimum_distance, collision_found, affected_link_index_list, affected_observed_point
        self.set_robot_position_in_obstacle_client(manip_joint_indices=manip_joint_indices,
                                                   target_position=target_position,
                                                   target_velocity=None,
                                                   capture_frame=True)

        minimum_distance = self._closest_point_maximum_relevant_distance

        for i in range(len(self._obstacle_list)):
            for link_index in self._obstacle_list[i].observed_links:
                if self._check_braking_trajectory_closest_points and \
                        self._links[link_index].closest_point_active:
                    pos_obs, pos_rob, distance = self._compute_closest_points(
                        p.getClosestPoints(bodyA=self._obstacle_list[i].id,
                                           bodyB=self._robot_scene.robot_id,
                                           distance=self._closest_point_maximum_relevant_distance,
                                           linkIndexA=self._obstacle_list[
                                               i].last_link,
                                           linkIndexB=link_index,
                                           physicsClientId=self._obstacle_client_id))

                    safety_distance = self._links[link_index].closest_point_safety_distance

                    if distance is not None:

                        if distance < minimum_distance:
                            minimum_distance = distance

                        if distance < safety_distance:
                            collision_found = True
                            affected_link_index_list = [link_index]
                            return minimum_distance, collision_found, affected_link_index_list, None

                if self._check_braking_trajectory_observed_points \
                        and len(self._links[link_index].observed_points) > 0:
                    for k in range(len(self._links[link_index].observed_points)):
                        if self._links[link_index].observed_points[k].is_active:
                            pos_rob = self._links[link_index].observed_points[k].get_position(actual=None)
                            pos_obs = self._obstacle_list[i].get_position(actual=False, pos_rob=pos_rob)

                            distance = self._compute_distance(pos_obs, pos_rob,
                                                              radius_a=self._obstacle_list[
                                                                  i].bounding_sphere_radius,
                                                              radius_b=self._links[link_index].observed_points[
                                                                  k].bounding_sphere_radius)
                            safety_distance = self._links[link_index].observed_points[k].safety_distance

                            if distance < minimum_distance:
                                minimum_distance = distance

                            if distance < safety_distance:
                                collision_found = True
                                affected_observed_point = [link_index, k]
                                return minimum_distance, collision_found, [], affected_observed_point

        for i in range(len(self._links)):
            for j in range(len(self._links[i].self_collision_links)):
                if (self._links[i].closest_point_active or self._links[
                    self._links[i].self_collision_links[j]].closest_point_active) \
                        and self._check_braking_trajectory_closest_points:
                    pos_rob_a, pos_rob_b, distance = self._compute_closest_points(
                        p.getClosestPoints(bodyA=self._robot_scene.robot_id,
                                           bodyB=self._robot_scene.robot_id,
                                           distance=self._closest_point_maximum_relevant_distance,
                                           linkIndexA=i,
                                           linkIndexB=self._links[i].self_collision_links[j],
                                           physicsClientId=self._obstacle_client_id))

                    safety_distance = self._links[i].closest_point_safety_distance

                    if distance is not None:
                        if distance < minimum_distance:
                            minimum_distance = distance

                        if distance < safety_distance:
                            collision_found = True
                            affected_link_index_list = [i, self._links[i].self_collision_links[j]]
                            return minimum_distance, collision_found, affected_link_index_list, None

        return minimum_distance, False, [], None

    def _check_if_braking_trajectory_is_torque_limited(self, time_step_counter=0):

        braking_time_step = 0

        self.set_robot_position_in_obstacle_client(target_position=self._last_actual_position,
                                                   target_velocity=self._last_actual_velocity)
        '''
        self.set_robot_position_in_obstacle_client(target_position=self._actual_position,
                                                   target_velocity=self._actual_velocity)
        '''
        self._robot_scene.set_motor_control(target_positions=self._braking_trajectory['position'][braking_time_step],
                                            target_velocities=self._braking_trajectory['velocity'][braking_time_step],
                                            physics_client_id=self._obstacle_client_id)

        for i in range(1):
            p.stepSimulation(physicsClientId=self._obstacle_client_id)

        '''
        actual_position, actual_velocity = \
            self._robot_scene.get_actual_joint_position_and_velocity(physics_client_id=self._obstacle_client_id)
        print("Actual pos diff: ", self._actual_position - actual_position)
        print("Actual vel diff: ", self._actual_velocity - actual_velocity)
        '''

        affected_link_index_list = []
        maximum_rel_torque = None

        time_since_start = np.linspace(self._trajectory_time_step / self._simulation_steps_per_action,
                                       self._trajectory_time_step,
                                       self._simulation_steps_per_action)

        if self._check_braking_trajectory_collisions:
            extra_time_steps = 0  # no additional time steps are required as a certain safety distance is guaranteed
        else:
            extra_time_steps = 1  # check additional time steps since the actual values lag behind the setpoints

        robot_stopped = False

        while not robot_stopped:
            interpolated_position_batch = \
                interpolate_position_batch(self._braking_trajectory['acceleration'][braking_time_step],
                                           self._braking_trajectory['acceleration'][braking_time_step + 1],
                                           self._braking_trajectory['velocity'][braking_time_step],
                                           self._braking_trajectory['position'][braking_time_step],
                                           time_since_start, self._trajectory_time_step)

            if self._robot_scene.use_controller_target_velocities:
                interpolated_velocity_batch = interpolate_velocity_batch(
                    self._braking_trajectory['acceleration'][braking_time_step],
                    self._braking_trajectory['acceleration'][braking_time_step + 1],
                    self._braking_trajectory['velocity'][braking_time_step],
                    time_since_start, self._trajectory_time_step)
            else:
                interpolated_velocity_batch = None
            for m in range(self._simulation_steps_per_action):
                self._robot_scene.set_motor_control(target_positions=interpolated_position_batch[m],
                                                    target_velocities=interpolated_velocity_batch[m]
                                                    if interpolated_velocity_batch is not None else None,
                                                    physics_client_id=self._obstacle_client_id)
                p.stepSimulation(physicsClientId=self._obstacle_client_id)
                actual_joint_torques = self._robot_scene.get_actual_joint_torques(
                    physics_client_id=self._obstacle_client_id)
                normalized_joint_torques = normalize(actual_joint_torques, self._torque_limits)
                normalized_joint_torques_abs = np.abs(normalized_joint_torques)
                maximum_rel_torque = np.max(normalized_joint_torques_abs)
                if maximum_rel_torque > self._braking_trajectory_maximum_rel_torque:
                    self._braking_trajectory_maximum_rel_torque = maximum_rel_torque

                joint_torque_exceeded = normalized_joint_torques_abs > 0.98

                if np.any(joint_torque_exceeded):
                    affected_link_index_list = np.array(self._manip_joint_indices)[joint_torque_exceeded]
                    return False, affected_link_index_list

            self._braking_trajectory['max_torque'].append(maximum_rel_torque)
            # max torque at m == self._simulation_steps_per_action - 1 -> end of the time step

            braking_time_step = braking_time_step + 1

            if not self._check_braking_trajectory_collisions:
                # compute the target acceleration for the next decision step
                robot_stopped, braking_timeout = \
                    self._compute_next_braking_trajectory_time_step(start_position=interpolated_position_batch[-1],
                                                                    extend_trajectory=extra_time_steps > 0)
                if robot_stopped and extra_time_steps > 0:
                    extra_time_steps = extra_time_steps - 1
                    robot_stopped = False

                if braking_timeout:
                    return False, affected_link_index_list
            else:
                # braking trajectory already calculated
                robot_stopped = braking_time_step > self._braking_trajectory_length
                if robot_stopped and extra_time_steps > 0:
                    self._compute_next_braking_trajectory_time_step(start_position=interpolated_position_batch[-1],
                                                                    extend_trajectory=True)
                    extra_time_steps = extra_time_steps - 1
                    robot_stopped = False

        return True, affected_link_index_list

    def _compute_next_braking_trajectory_time_step(self, start_position, extend_trajectory=False, braking_timeout=2.0):

        if self._braking_duration > braking_timeout:
            self._braking_timeout = True
            return False, self._braking_timeout

        start_velocity = interpolate_velocity(self._braking_trajectory['acceleration'][-2],
                                              self._braking_trajectory['acceleration'][-1],
                                              self._braking_trajectory['velocity'][-1],
                                              self._trajectory_time_step, self._trajectory_time_step)
        start_acceleration = self._braking_trajectory['acceleration'][-1]

        joint_acc_min, joint_acc_max = self._acc_range_function(start_position=start_position,
                                                                start_velocity=start_velocity,
                                                                start_acceleration=
                                                                start_acceleration)

        end_acceleration, robot_stopped = self._acc_braking_function(start_velocity=start_velocity,
                                                                     start_acceleration=start_acceleration,
                                                                     next_acc_min=joint_acc_min,
                                                                     next_acc_max=joint_acc_max,
                                                                     index=0)

        if not robot_stopped or extend_trajectory:
            self._braking_trajectory['acceleration'].append(end_acceleration)
            self._braking_trajectory['velocity'].append(start_velocity)
            self._braking_trajectory['position'].append(start_position)

        return robot_stopped, False

    def _interpolate_position(self, start_acceleration, end_acceleration, start_velocity, start_position,
                              time_since_start):
        interpolated_position = start_position + start_velocity * time_since_start + \
                                0.5 * start_acceleration * time_since_start ** 2 + \
                                1 / 6 * ((end_acceleration - start_acceleration)
                                         / self._trajectory_time_step) * time_since_start ** 3

        return interpolated_position

    def _interpolate_velocity(self, start_acceleration, end_acceleration, start_velocity, time_since_start):

        interpolated_velocity = start_velocity + start_acceleration * time_since_start + \
                                0.5 * ((end_acceleration - start_acceleration) /
                                       self._trajectory_time_step) * time_since_start ** 2

        return interpolated_velocity

    def _interpolate_acceleration(self, start_acceleration, end_acceleration, time_since_start):
        interpolated_acceleration = start_acceleration + ((end_acceleration - start_acceleration) /
                                                          self._trajectory_time_step) * time_since_start

        return interpolated_acceleration

    def _normalize(self, value, value_range):
        normalized_value = -1 + 2 * (value - value_range[0]) / (value_range[1] - value_range[0])
        return normalized_value

    @property
    def num_links(self):
        return len(self._links)

    @property
    def _braking_duration(self):
        return self._braking_trajectory_length * self._trajectory_time_step

    @property
    def _braking_trajectory_length(self):
        return len(self._braking_trajectory['position']) - 1  # only the "braking" part (first time step excluded)

    @property
    def link_names(self):
        link_names = [link.name for link in self._links]
        return link_names

    @property
    def debug_line_obstacle(self):
        return self._debug_line_obstacle

    @debug_line_obstacle.setter
    def debug_line_obstacle(self, val):
        self._debug_line_obstacle = val

    @property
    def debug_line_link(self):
        return self._debug_line_link

    @debug_line_link.setter
    def debug_line_link(self, val):
        self._debug_line_link = val

    @property
    def debug_line_point(self):
        return self._debug_line_point

    @debug_line_point.setter
    def debug_line_point(self, val):
        self._debug_line_point = val


class ObstacleBase:
    def __init__(self,
                 name=None,
                 observed_links=None,
                 num_observed_points_per_link=None,
                 *vargs,
                 **kwargs):

        if num_observed_points_per_link is None:
            num_observed_points_per_link = []
        if observed_links is None:
            observed_links = []
        self._observed_links = observed_links
        if len(self._observed_links) != len(num_observed_points_per_link):
            raise ValueError("observed Points per Link not specified")

        self._link_data = []
        for i in range(len(observed_links)):
            self._link_data.append(LinkData(num_observed_points=num_observed_points_per_link[i]))

        self._name = name

    def get_link_index(self, link_number):
        for i in range(len(self._observed_links)):
            if self._observed_links[i] == link_number:
                return i
        raise ValueError("Desired link is not observed")

    @property
    def observed_links(self):
        return self._observed_links

    @property
    def name(self):
        return self._name

    @property
    def link_data(self):
        return self._link_data


class ObstacleSim(ObstacleBase):
    def __init__(self,
                 pos=(0, 0, 0),
                 orn=(0.0, 0.0, 0.0, 1.0),
                 create_collision_shape=True,
                 is_static=False,
                 shape=p.GEOM_SPHERE,
                 urdf_file_name=None,
                 radius=None,
                 half_extents=None,
                 plane_points=None,
                 color=(0, 0, 1, 0.5),
                 simulation_client_id=None,
                 obstacle_client_id=None,
                 use_real_robot=False,
                 num_clients=1,
                 is_target=False,
                 plane_collision_shape_factor=1,
                 *vargs,
                 **kwargs):

        super().__init__(*vargs, **kwargs)
        default_orn = p.getQuaternionFromEuler([0, 0, 0])

        self._shape = shape
        self._bounding_sphere_radius = 0
        self._is_static = is_static
        self._base_pos = pos
        self._base_orn = orn
        self._target_pos = [0, 0, 0]
        self._target_vel = [0, 0, 0]
        self._target_acc = [0, 0, 0]

        self._position_actual = None
        self._position_set = None
        self._orn_actual = None
        self._orn_set = None
        self._is_target = is_target

        self._num_clients = num_clients
        self._simulation_client_id = simulation_client_id
        self._obstacle_client_id = obstacle_client_id
        self._use_real_robot = use_real_robot

        for i in range(self._num_clients):
            if urdf_file_name is None:
                if shape == p.GEOM_SPHERE:
                    if radius is None:
                        raise ValueError("Radius required")
                    self._bounding_sphere_radius = radius
                    if create_collision_shape:
                        shape_collision = p.createCollisionShape(shapeType=p.GEOM_SPHERE, radius=radius,
                                                                 physicsClientId=i)
                    else:
                        shape_collision = -1

                    shape_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=radius,
                                                       rgbaColor=color, physicsClientId=i)

                if shape == p.GEOM_BOX:
                    if half_extents is None:
                        raise ValueError("half_extents required")
                    if create_collision_shape:
                        shape_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extents,
                                                                 physicsClientId=i)
                    else:
                        shape_collision = -1

                    shape_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extents,
                                                       rgbaColor=color, physicsClientId=i)

                if shape == p.GEOM_PLANE:
                    plane_points = np.array(plane_points)
                    plane_x = np.linalg.norm(plane_points[1] - plane_points[0])
                    plane_y = np.linalg.norm(plane_points[2] - plane_points[0])
                    self._base_pos = plane_points[0] + 0.5 * (plane_points[1] - plane_points[0]) + \
                                     0.5 * (plane_points[2] - plane_points[0])
                    half_extents_visual = [plane_x / 2, plane_y / 2, 0.0025]
                    shape_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extents_visual,
                                                       rgbaColor=color, physicsClientId=i)
                    half_extents_collision = [plane_x / 2 * plane_collision_shape_factor,
                                              plane_y / 2 * plane_collision_shape_factor, 0.0025]
                    if create_collision_shape:
                        shape_collision = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                                 halfExtents=half_extents_collision,
                                                                 physicsClientId=i)
                    else:
                        shape_collision = -1

                    self._plane_normal = np.cross(plane_points[1] - plane_points[0], plane_points[2] - plane_points[0])
                    self._plane_constant = np.dot(plane_points[0], self._plane_normal)

                if shape == p.GEOM_PLANE or self._is_static:
                    self._last_link = -1
                    self._is_static = True

                    self.id = p.createMultiBody(baseMass=0,
                                                basePosition=self._base_pos,
                                                baseOrientation=orn,
                                                baseVisualShapeIndex=shape_visual,
                                                baseCollisionShapeIndex=shape_collision,
                                                physicsClientId=i)
                else:
                    self._last_link = 2
                    self.id = p.createMultiBody(baseMass=0,
                                                basePosition=self._base_pos,
                                                baseOrientation=orn,
                                                linkMasses=[1, 1, 1],
                                                linkVisualShapeIndices=[-1, -1, shape_visual],
                                                linkCollisionShapeIndices=[-1, -1, shape_collision],
                                                linkPositions=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                                linkOrientations=[default_orn, default_orn, default_orn],
                                                linkParentIndices=[0, 1, 2],
                                                linkJointTypes=[p.JOINT_PRISMATIC, p.JOINT_PRISMATIC,
                                                                p.JOINT_PRISMATIC],
                                                linkJointAxis=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                                                linkInertialFramePositions=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                                                linkInertialFrameOrientations=[default_orn, default_orn, default_orn],
                                                physicsClientId=i
                                                )
            else:
                self._is_static = True
                self._last_link = -1
                if not urdf_file_name.endswith(".urdf"):
                    urdf_file_name = urdf_file_name + ".urdf"
                module_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
                urdf_dir = os.path.join(module_dir, "description", "urdf", "obstacles")
                self.id = p.loadURDF(os.path.join(urdf_dir, urdf_file_name), basePosition=self._base_pos,
                                     useFixedBase=True, physicsClientId=i)
                p.changeVisualShape(self.id, -1, rgbaColor=color)

        if not (shape == p.GEOM_PLANE or self._is_static):
            for i in range(3):
                p.setJointMotorControl2(self.id, i,
                                        p.POSITION_CONTROL,
                                        targetPosition=self._target_pos[i],
                                        targetVelocity=0,
                                        maxVelocity=0,
                                        positionGain=0.1,
                                        velocityGain=1)

    def update(self):
        pass

    def get_local_position(self, world_position, actual=True):
        return np.array(world_position) - np.array(self.get_position(actual=actual))

    def make_invisible(self):
        for i in range(self._num_clients):
            p.changeVisualShape(self.id, -1, rgbaColor=[1, 1, 1, 0], physicsClientId=i)

    @property
    def target_pos(self):
        return self._target_pos

    @property
    def is_target(self):
        return self._is_target

    @property
    def target_vel(self):
        return self._target_vel

    @property
    def target_acc(self):
        return self._target_acc

    @property
    def is_static(self):
        return self._is_static

    def get_position_set(self):
        return self._position_set

    def get_orn(self, actual=True):
        if actual:
            if self._orn_actual is None:
                self.get_position(actual=actual)
            return self._orn_actual
        else:
            if self._orn_set is None:
                self.get_position(actual=actual)
            return self._orn_set

    def get_position(self, actual=True, pos_rob=None):
        if self._shape is not p.GEOM_PLANE:
            if actual:
                if self._position_actual is None:
                    if self._is_static:
                        self._position_actual, self._orn_actual = self._base_pos, self._base_orn
                    else:
                        if self._simulation_client_id is not None and not self._use_real_robot:
                            link_state = p.getLinkState(self.id, 2, computeLinkVelocity=False,
                                                        computeForwardKinematics=True,
                                                        physicsClientId=self._simulation_client_id)
                            self._position_actual = link_state[4]
                            self._orn_actual = link_state[5]
                        else:
                            raise NotImplementedError("Actual obstacle position not implemented for real robots")
                return self._position_actual
            else:
                if self._position_set is None:
                    if self._is_static:
                        self._position_set, self._orn_set = self._base_pos, self._base_orn
                    else:
                        link_state = p.getLinkState(self.id, 2, computeLinkVelocity=False,
                                                    computeForwardKinematics=True,
                                                    physicsClientId=self._obstacle_client_id)
                        self._position_set = link_state[4]
                        self._orn_set = link_state[5]
                return self._position_set
        else:
            pos_rob = np.array(pos_rob)
            x = (self._plane_constant - np.dot(pos_rob, self._plane_normal)) / (np.linalg.norm(self._plane_normal) ** 2)
            return pos_rob + x * self._plane_normal

    def reset(self):
        self.clear_previous_timestep()
        for i in range(len(self._link_data)):
            self._link_data[i].reset()

        if not self._is_static:
            raise NotImplementedError("Reset function for non static obstacles not implemented")

    def clear_previous_timestep(self):
        if not self._is_static:
            self._position_actual = None
            self._position_set = None
            self._orn_actual = None
            self._orn_set = None

    @property
    def last_link(self):
        return self._last_link

    @property
    def bounding_sphere_radius(self):
        return self._bounding_sphere_radius


class LinkPointBase(object):
    def __init__(self,
                 name="tbd",
                 offset=(0, 0, 0),
                 bounding_sphere_radius=0.0,
                 active=False,
                 safety_distance=0.00,
                 visualize_bounding_sphere=False,
                 default_bounding_sphere_color=(0, 1, 0, 0.5),
                 num_clients=1,
                 *vargs,
                 **kwargs):
        self._name = name
        self._offset = offset
        self._bounding_sphere_radius = bounding_sphere_radius
        self._default_bounding_sphere_color = default_bounding_sphere_color
        self._bounding_sphere_color = self._default_bounding_sphere_color
        self._active = active
        self._link_object = None
        self._visualize_bounding_sphere = visualize_bounding_sphere
        self._safetyDistance = safety_distance
        self._num_clients = num_clients

        if self._visualize_bounding_sphere and self._bounding_sphere_radius > 0:
            for i in range(self._num_clients):
                shape_visual = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=self._bounding_sphere_radius,
                                                   rgbaColor=self._default_bounding_sphere_color, physicsClientId=i)

                self._bounding_sphere_id = p.createMultiBody(baseMass=0,
                                                             basePosition=[0, 0, 0],
                                                             baseOrientation=[0, 0, 0, 1],
                                                             baseVisualShapeIndex=shape_visual,
                                                             physicsClientId=i)
        else:
            self._bounding_sphere_id = None

    def update_bounding_sphere_position(self, actual=True):
        if self._bounding_sphere_id is not None:
            pos = self.get_position(actual=actual)
            def_orn = p.getQuaternionFromEuler([0, 0, 0])
            for i in range(self._num_clients):
                p.resetBasePositionAndOrientation(bodyUniqueId=self._bounding_sphere_id, posObj=pos, ornObj=def_orn,
                                                  physicsClientId=i)

    def set_bounding_sphere_color(self, rgba_color=None):
        if self._bounding_sphere_id is not None:
            if rgba_color is None:
                rgba_color = self._default_bounding_sphere_color
            if rgba_color != self._bounding_sphere_color:
                self._bounding_sphere_color = rgba_color
                if self._link_object.simulation_client_id is not None:
                    p.changeVisualShape(self._bounding_sphere_id, -1, rgbaColor=self._bounding_sphere_color,
                                        physicsClientId=self._link_object.simulation_client_id)

    def get_position(self, actual=None, return_orn=False):

        def_orn = p.getQuaternionFromEuler([0, 0, 0])
        pos, orn = p.multiplyTransforms(positionA=self._link_object.get_position(actual=actual),
                                        orientationA=self._link_object.get_orn(actual=actual),
                                        positionB=self._offset,
                                        orientationB=def_orn)

        if return_orn:
            return pos, orn
        else:
            return pos

    @property
    def name(self):
        return self._name

    @property
    def is_active(self):
        return self._active

    @property
    def safety_distance(self):
        return self._safetyDistance

    @property
    def offset(self):
        return self._offset

    @property
    def link_object(self):
        return self._link_object

    @link_object.setter
    def link_object(self, val):
        self._link_object = val

    @property
    def bounding_sphere_radius(self):
        return self._bounding_sphere_radius


class LinkBase(object):
    def __init__(self,
                 name=None,
                 observe_closest_point=True,
                 closest_point_active=False,
                 closest_point_safety_distance=0.1,
                 observed_points=None,
                 index=None,
                 robot_id=None,
                 robot_index=None,
                 self_collision_links=None,
                 default_color=None,
                 simulation_client_id=None,
                 obstacle_client_id=None,
                 use_real_robot=False,
                 set_robot_position_in_obstacle_client_function=None,
                 is_obstacle_client_at_other_position_function=None,
                 *vargs,
                 **kwargs):

        if default_color is None:
            default_color = [0.9, 0.9, 0.9, 1]
        if self_collision_links is None:
            self_collision_links = []
        if observed_points is None:
            observed_points = []

        self._observe_closest_point = observe_closest_point
        self._closest_point_active = closest_point_active
        self._closest_point_safety_distance = closest_point_safety_distance

        self._simulation_client_id = simulation_client_id
        self._obstacle_client_id = obstacle_client_id
        self._use_real_robot = use_real_robot

        if self._closest_point_active:
            self._observe_closest_point = True
        self._name = name
        self._observed_points = observed_points
        for i in range(len(self._observed_points)):
            self._observed_points[i].link_object = self
        self._index = index
        self._robot_id = robot_id
        self._robot_index = robot_index

        self._position_actual = None
        self._position_set = None
        self._orn_set = None
        self._orn_actual = None
        self._position_other = None
        self._orn_other = None
        self._self_collision_links = self_collision_links
        self._self_collision_data = SelfCollisionData(num_self_collision_links=len(self._self_collision_links))

        self._color = None
        self._default_color = default_color

        self._set_robot_position_in_obstacle_client_function = set_robot_position_in_obstacle_client_function
        self._is_obstacle_client_at_other_position_function = is_obstacle_client_at_other_position_function
        self.set_color(rgba_color=None, obstacle_client=True)

    def get_local_position(self, world_position, actual=True):
        def_orn = p.getQuaternionFromEuler([0, 0, 0])
        com_pos_inv, com_orn_inv = p.invertTransform(position=self.get_position(actual=actual),
                                                     orientation=self.get_orn(actual=actual))
        pos, _ = p.multiplyTransforms(positionA=com_pos_inv,
                                      orientationA=com_orn_inv,
                                      positionB=world_position,
                                      orientationB=def_orn)

        return pos

    def get_orn(self, actual=True):
        if actual is None:
            if self._orn_other is None:
                self.get_position(actual=actual)
            return self._orn_other
        else:
            if actual:
                if self._orn_actual is None:
                    self.get_position(actual=actual)
                return self._orn_actual
            else:
                if self._orn_set is None:
                    self.get_position(actual=actual)
                return self._orn_set

    def get_position(self, actual=None):
        if actual is None:
            if self._position_other is None:
                link_state = p.getLinkState(bodyUniqueId=self._robot_id, linkIndex=self._index,
                                            computeLinkVelocity=False, computeForwardKinematics=True,
                                            physicsClientId=self._obstacle_client_id)
                self._position_other = link_state[4]
                self._orn_other = link_state[5]

            return self._position_other

        else:
            if actual:
                if self._position_actual is None:
                    self._set_robot_position_in_obstacle_client_function(set_to_actual_values=True)
                    link_state = p.getLinkState(bodyUniqueId=self._robot_id, linkIndex=self._index,
                                                computeLinkVelocity=False,
                                                computeForwardKinematics=True, physicsClientId=self._obstacle_client_id)
                    self._position_actual = link_state[4]
                    self._orn_actual = link_state[5]

                return self._position_actual

            else:
                # set point
                if self._position_set is None:
                    self._set_robot_position_in_obstacle_client_function(set_to_setpoints=True)
                    link_state = p.getLinkState(bodyUniqueId=self._robot_id, linkIndex=self._index,
                                                computeLinkVelocity=False,
                                                computeForwardKinematics=True, physicsClientId=self._obstacle_client_id)
                    self._position_set = link_state[4]
                    self._orn_set = link_state[5]
                return self._position_set

    def set_color(self, rgba_color, obstacle_client=False):
        if rgba_color is None:
            rgba_color = self._default_color
        if rgba_color != self._color:
            self._color = rgba_color
            if self._simulation_client_id is not None:
                p.changeVisualShape(self._robot_id, self._index, -1,
                                    rgbaColor=rgba_color, physicsClientId=self._simulation_client_id)
            if obstacle_client and self._obstacle_client_id is not None:
                p.changeVisualShape(self._robot_id, self._index, -1,
                                    rgbaColor=rgba_color, physicsClientId=self._obstacle_client_id)

    def clear_previous_timestep(self):
        self._position_actual = None
        self._position_set = None
        self._orn_set = None
        self._orn_actual = None
        self._position_other = None
        self._orn_other = None

    def clear_other_position_and_orn(self):
        self._position_other = None
        self._orn_other = None

    def reset(self):
        self.set_color(None)
        self._self_collision_data.reset()

    @property
    def self_collision_links(self):
        return self._self_collision_links

    @property
    def self_collision_data(self):
        return self._self_collision_data

    @property
    def robot_index(self):
        return self._robot_index

    @property
    def observe_closest_point(self):
        return self._observe_closest_point

    @property
    def closest_point_active(self):
        return self._closest_point_active

    @property
    def closest_point_safety_distance(self):
        return self._closest_point_safety_distance

    @property
    def observed_points(self):
        return self._observed_points

    @property
    def num_observed_points(self):
        return len(self._observed_points)

    @property
    def name(self):
        return self._name

    @property
    def index(self):
        return self._index

    @property
    def simulation_client_id(self):
        return self._simulation_client_id


class SelfCollisionData(object):
    def __init__(self,
                 num_self_collision_links):
        self._num_self_collision_links = num_self_collision_links
        self._closest_point_distance_actual = None
        self._closest_point_distance_set = None
        self.reset()

    def reset(self):
        self._closest_point_distance_actual = [[] for _ in range(self._num_self_collision_links)]
        self._closest_point_distance_set = [[] for _ in range(self._num_self_collision_links)]

    def export_metrics(self, export_link_pair=None):
        export_dict = {}
        if export_link_pair is None:
            export_link_pair = [True] * len(self._closest_point_distance_actual)
        export_dict['closest_point_distance_actual_min'] = [np.min(
            self._closest_point_distance_actual[i]) if self._closest_point_distance_actual[i] else None
                                                            for i in range(len(self._closest_point_distance_actual))
                                                            if export_link_pair[i]]
        export_dict['closest_point_distance_set_min'] = [np.min(
            self._closest_point_distance_set[i]) if self._closest_point_distance_set[i] else None for i in
                                                         range(len(self._closest_point_distance_set))
                                                         if export_link_pair[i]]

        return export_dict

    @property
    def closest_point_distance_actual(self):
        return self._closest_point_distance_actual

    @property
    def closest_point_distance_set(self):
        return self._closest_point_distance_set


class LinkData(object):
    def __init__(self,
                 num_observed_points,
                 *vargs,
                 **kwargs):
        self._num_observed_points = num_observed_points
        self._closest_point_distance_actual = None
        self._closest_point_distance_set = None
        self._closest_point_velocity_set = None
        self._observed_point_distance_actual = None
        self._observed_point_distance_set = None
        self._observed_point_velocity_set = None
        self.reset()

    def reset(self):
        self._closest_point_distance_actual = []
        self._closest_point_distance_set = []
        self._closest_point_velocity_set = []
        self._observed_point_distance_actual = [[] for _ in range(self._num_observed_points)]
        self._observed_point_distance_set = [[] for _ in range(self._num_observed_points)]
        self._observed_point_velocity_set = [[] for _ in range(self._num_observed_points)]

    def export_metrics(self):
        export_dict = {}
        export_dict['closest_point_distance_actual_min'] = np.min(
            self._closest_point_distance_actual) if self._closest_point_distance_actual else None
        export_dict['closest_point_distance_set_min'] = np.min(
            self._closest_point_distance_set) if self._closest_point_distance_set else None
        export_dict['observed_point_distance_actual_min'] = [np.min(
            self._observed_point_distance_actual[i]) if self._observed_point_distance_actual[i] else None for i in
                                                             range(len(self._observed_point_distance_actual))]
        export_dict['observed_point_distance_set_min'] = [np.min(
            self._observed_point_distance_set[i]) if self._observed_point_distance_set[i] else None for i in
                                                          range(len(self._observed_point_distance_set))]

        return export_dict

    @property
    def closest_point_distance_actual(self):
        return self._closest_point_distance_actual

    @property
    def closest_point_velocity_set(self):
        return self._closest_point_velocity_set

    @property
    def closest_point_distance_set(self):
        return self._closest_point_distance_set

    @property
    def observed_point_velocity_set(self):
        return self._observed_point_velocity_set

    @property
    def observed_point_distance_actual(self):
        return self._observed_point_distance_actual

    @property
    def observed_point_distance_set(self):
        return self._observed_point_distance_set

    @property
    def num_observed_points(self):
        return len(self._observed_point_distance_actual)
