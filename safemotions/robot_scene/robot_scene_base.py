# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import inspect
import logging
import os
import re

import numpy as np
import pybullet as p

from safemotions.robot_scene.obstacle_torque_prevention import ObstacleWrapperSim


class RobotSceneBase(object):
    MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
    URDF_DIR = os.path.join(MODULE_DIR, "description", "urdf")
    JOINT_LIMITS_SAFETY_BUFFER = 0.035
    MAX_ACCELERATIONS = [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]
    MAX_JERK = [7500, 3750, 5000, 6250, 7500, 10000, 10000]

    def __init__(self,
                 simulation_client_id=None,
                 simulation_time_step=None,
                 obstacle_client_id=None,
                 trajectory_time_step=None,
                 use_real_robot=False,
                 robot_scene=0,
                 obstacle_scene=0,
                 observed_link_point_scene=0,
                 log_obstacle_data=False,
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
                 no_self_collision=False,
                 target_link_name="iiwa_link_7",
                 target_link_offset=None,
                 pos_limit_factor=1,
                 vel_limit_factor=1,
                 acc_limit_factor=1,
                 jerk_limit_factor=1,
                 torque_limit_factor=1,
                 **kwargs):

        if target_link_offset is None:
            target_link_offset = [0, 0, 0.0]
        self._simulation_client_id = simulation_client_id
        self._simulation_time_step = simulation_time_step
        self._obstacle_client_id = obstacle_client_id
        self._use_real_robot = use_real_robot
        self._trajectory_time_step = trajectory_time_step

        self._num_clients = 0
        if self._simulation_client_id is not None:
            self._num_clients += 1
        if self._obstacle_client_id is not None:
            self._num_clients += 1

        self._no_self_collision = no_self_collision

        self._num_robots = None
        self._robot_scene = robot_scene

        robot_urdf = None
        if robot_scene == 0:
            self._num_robots = 1
            robot_urdf = "one_robot"

        if robot_scene == 1:
            self._num_robots = 2
            robot_urdf = "two_robots"

        if robot_scene == 2:
            self._num_robots = 3
            robot_urdf = "three_robots"

        for i in range(self._num_clients):
            if self._no_self_collision:
                self._robot_id = p.loadURDF(os.path.join(self.URDF_DIR, robot_urdf + ".urdf"), useFixedBase=True,
                                            physicsClientId=i)
            else:
                self._robot_id = p.loadURDF(os.path.join(self.URDF_DIR, robot_urdf + ".urdf"), useFixedBase=True,
                                            flags=p.URDF_USE_SELF_COLLISION,
                                            physicsClientId=i)

        if self._simulation_client_id is not None:
            p.changeVisualShape(self._robot_id, 0, -1, rgbaColor=[0.3, 0.3, 0.3, 1])  # set table color to gray

        joint_lower_limits, joint_upper_limits, force_limits, velocity_limits = [], [], [], []

        if self._num_clients != 0:
            self._manip_joint_indices, self._manip_joint_indices_per_robot = self._get_manip_joint_indices()
            self._num_manip_joints = len(self._manip_joint_indices)
            self._link_name_list = []

            for i in range(p.getNumJoints(self._robot_id)):
                joint_info = p.getJointInfo(self._robot_id, i)
                self._link_name_list.append(
                    str(joint_info[12])[2:-1])  # link name is loaded as  b'linkname' -> extract linkname

            for i in self._manip_joint_indices:
                joint_infos = p.getJointInfo(self._robot_id, i)
                joint_lower_limits.append(joint_infos[8] + self.JOINT_LIMITS_SAFETY_BUFFER)
                joint_upper_limits.append(joint_infos[9] - self.JOINT_LIMITS_SAFETY_BUFFER)
                force_limits.append(joint_infos[10])
                velocity_limits.append(joint_infos[11])
            self._initial_joint_lower_limits = np.array(joint_lower_limits)
            self._initial_joint_upper_limits = np.array(joint_upper_limits)
            self._initial_max_torques = np.array(force_limits)
            self._initial_max_velocities = np.array(velocity_limits)
            self._initial_max_accelerations = np.array(self.MAX_ACCELERATIONS * self._num_robots)
            self._initial_max_jerk = np.array(self.MAX_JERK * self._num_robots)

            self._obstacle_wrapper = \
                ObstacleWrapperSim(robot_scene=self,
                                   simulation_client_id=self._simulation_client_id,
                                   simulation_time_step=self._simulation_time_step,
                                   obstacle_client_id=self._obstacle_client_id,
                                   use_real_robot=self._use_real_robot,
                                   obstacle_scene=obstacle_scene,
                                   observed_link_point_scene=observed_link_point_scene,
                                   log_obstacle_data=log_obstacle_data,
                                   link_name_list=self._link_name_list,
                                   manip_joint_indices=self._manip_joint_indices,
                                   use_braking_trajectory_method=use_braking_trajectory_method,
                                   check_braking_trajectory_torque_limits=check_braking_trajectory_torque_limits,
                                   collision_check_time=collision_check_time,
                                   check_braking_trajectory_observed_points=check_braking_trajectory_observed_points,
                                   check_braking_trajectory_closest_points=check_braking_trajectory_closest_points,
                                   closest_point_safety_distance=closest_point_safety_distance,
                                   observed_point_safety_distance=observed_point_safety_distance,
                                   use_target_points=use_target_points,
                                   target_point_cartesian_range_scene=target_point_cartesian_range_scene,
                                   target_point_radius=target_point_radius,
                                   target_point_sequence=target_point_sequence,
                                   target_point_reached_reward_bonus=target_point_reached_reward_bonus,
                                   target_link_name=target_link_name,
                                   target_link_offset=target_link_offset,
                                   visualize_bounding_spheres=visualize_bounding_spheres,
                                   )
        else:
            self._manip_joint_indices = None
            self._manip_joint_indices_per_robot = None
            self._num_manip_joints = None
            self._obstacle_wrapper = None
            self._link_name_list = None

        for i in range(self._num_clients):
            half_extents = [10, 10, 0.0025]
            shape_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extents,
                                               rgbaColor=(0.9, 0.9, 0.9, 1), physicsClientId=i)
            shape_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extents,
                                                     physicsClientId=i)

            self._plane_id = p.createMultiBody(baseMass=0, basePosition=[0, 0, -0.94],
                                               baseOrientation=(0.0, 0.0, 0.0, 1.0),
                                               baseVisualShapeIndex=shape_visual,
                                               baseCollisionShapeIndex=shape_collision,
                                               physicsClientId=i)

        self._joint_lower_limits = None
        self._joint_upper_limits = None
        self._max_velocities = None
        self._max_accelerations = None
        self._max_jerk_linear_interpolation = None
        self._max_torques = None

        self._pos_limit_factor = pos_limit_factor
        self._vel_limit_factor = vel_limit_factor
        self._acc_limit_factor = acc_limit_factor
        self._jerk_limit_factor = jerk_limit_factor
        self._torque_limit_factor = torque_limit_factor

        self._trajectory_index = -1

    def compute_actual_joint_limits(self):
        self._joint_lower_limits = list(np.array(self._initial_joint_lower_limits) * self._pos_limit_factor)
        self._joint_upper_limits = list(np.array(self._initial_joint_upper_limits) * self._pos_limit_factor)
        self._max_velocities = self._initial_max_velocities * self._vel_limit_factor
        self._max_accelerations = self._initial_max_accelerations * self._acc_limit_factor
        self._max_jerk_linear_interpolation = np.array([min(2 * self._max_accelerations[i] / self._trajectory_time_step,
                                                            self._initial_max_jerk[i]) * self._jerk_limit_factor
                                                        for i in range(len(self._max_accelerations))])
        self._max_torques = self._initial_max_torques * self._torque_limit_factor

        if self._obstacle_wrapper is not None:
            self._obstacle_wrapper.trajectory_time_step = self._trajectory_time_step
            self._obstacle_wrapper.torque_limits = np.array([-1 * self._max_torques, self._max_torques])

        logging.info("Pos limits: %s", self._joint_upper_limits)
        logging.info("Vel limits: %s", self._max_velocities)
        logging.info("Acc limits: %s", self._max_accelerations)
        logging.info("Jerk limits: %s", self._max_jerk_linear_interpolation)
        logging.info("Torque limits: %s", self._max_torques)

    def prepare_for_end_of_episode(self):
        # overwritten by reality_wrapper
        pass

    def prepare_for_start_of_episode(self):
        # overwritten by reality_wrapper
        pass

    @property
    def manip_joint_indices(self):
        return self._manip_joint_indices

    @property
    def num_manip_joints(self):
        return self._num_manip_joints

    @property
    def robot_id(self):
        return self._robot_id

    @property
    def num_robots(self):
        return self._num_robots

    @property
    def joint_lower_limits(self):
        return self._joint_lower_limits

    @property
    def joint_upper_limits(self):
        return self._joint_upper_limits

    @property
    def max_velocities(self):
        return self._max_velocities

    @property
    def max_accelerations(self):
        return self._max_accelerations

    @property
    def max_jerk_linear_interpolation(self):
        return self._max_jerk_linear_interpolation

    @property
    def max_torques(self):
        return self._max_torques

    @property
    def robot_scene_id(self):
        return self._robot_scene

    @property
    def num_clients(self):
        return self._num_clients

    @property
    def obstacle_wrapper(self):
        return self._obstacle_wrapper

    def get_manip_joint_indices_per_robot(self, robot_index):
        return self._manip_joint_indices_per_robot[robot_index]

    def get_actual_joint_positions(self):
        return [s[0] for s in p.getJointStates(self._robot_id, self._manip_joint_indices)]

    def _get_manip_joint_indices(self):
        joint_indices = []
        joint_indices_per_robot = [[] for _ in range(self._num_robots)]
        for i in range(p.getNumJoints(self._robot_id)):
            joint_info = p.getJointInfo(self._robot_id, i)
            q_index = joint_info[3]  # to distinguish fixed from moving joints
            if q_index > -1:
                joint_indices.append(i)
                if self._num_robots == 1:
                    joint_indices_per_robot[0].append(i)
                else:
                    link_name = str(joint_info[12])[2:-1]
                    if re.match('^.*_r[0-9]+$', link_name):
                        # e.g. extract 1 from linkname_r1
                        robot_index = int(link_name.rsplit('_', 1)[1][1:])
                        if robot_index >= self._num_robots:
                            raise ValueError("Found link name " + link_name + ", but expected " + str(
                                self._num_robots) + " robots only.")
                        else:
                            joint_indices_per_robot[robot_index].append(i)
                    else:
                        raise ValueError("Could not find a robot suffix like _r0 for link " + link_name)

        return tuple(joint_indices), joint_indices_per_robot

    def set_motor_control(self, target_positions, mode=p.POSITION_CONTROL, physics_client_id=0,
                          manip_joint_indices=None, **kwargs):
        # overwritten by real robot scene
        if manip_joint_indices is None:
            manip_joint_indices = self._manip_joint_indices
        p.setJointMotorControlArray(self._robot_id, manip_joint_indices,
                                    mode, targetPositions=target_positions,
                                    physicsClientId=physics_client_id)

    def get_actual_joint_torques(self, physics_client_id=0, manip_joint_indices=None):
        # overwritten by reality wrapper
        if manip_joint_indices is None:
            manip_joint_indices = self.manip_joint_indices
        return np.swapaxes(np.array(p.getJointStates(self._robot_id, manip_joint_indices,
                                                     physicsClientId=physics_client_id), dtype=object), 0, 1)[3]

    def get_link_names_for_multiple_robots(self, link_names, robot_indices=None):
        if isinstance(link_names, str):
            link_names = [link_names]

        if self._num_robots == 1:
            return link_names  # do nothing if there is one robot only

        link_names_multiple_robots = []

        if robot_indices is None:
            robot_indices = np.arange(self._num_robots)

        for i in range(len(robot_indices)):
            for j in range(len(link_names)):
                link_names_multiple_robots.append(link_names[j] + "_r" + str(robot_indices[i]))

        return link_names_multiple_robots

    def get_robot_identifier_from_link_name(self, link_name):
        # returns _r1 for a link called iiwa_link_4_r1 and "" for a link called iiwa_link_4
        if re.match('^.*_r[0-9]+$', link_name):
            link_identifier = "_" + link_name.rsplit('_', 1)[1]
            # split string from the right side and choose the last element
        else:
            link_identifier = ""
        return link_identifier

    def get_link_index_from_link_name(self, link_name):
        for i in range(len(self._link_name_list)):
            if self._link_name_list[i] == link_name:
                return i
        return -1

    def get_robot_index_from_link_name(self, link_name):
        # returns the robot index extracted from the link name, e.g. 1 for iiwa_link_4_r1
        # returns -1 if no link index is found and if multiple robots are in use, 0 otherwise
        if self._num_robots > 1:
            if re.match('^.*_r[0-9]+$', link_name):
                # e.g. extract 1 from linkname_r1
                return int(link_name.rsplit('_', 1)[1][1:])
            else:
                return -1
        else:
            return 0

    def disconnect(self):
        for i in range(self._num_clients):
            p.disconnect(physicsClientId=i)
