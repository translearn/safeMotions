# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import inspect
import logging
import os
os.environ["LC_NUMERIC"] = "en_US.UTF-8"  # avoid wrong parsing of urdf files caused by localization (, vs .)
import re
import numpy as np
import pybullet as p

from safemotions.robot_scene.collision_torque_limit_prevention import ObstacleWrapperSim


class RobotSceneBase(object):
    MODULE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
    URDF_DIR = os.path.join(MODULE_DIR, "description", "urdf")
    JOINT_LIMITS_SAFETY_BUFFER_IIWA = 0.035
    JOINT_LIMITS_SAFETY_BUFFER_ARMAR = 0
    MAX_ACCELERATIONS_IIWA = [15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0]
    MAX_JERK_IIWA = [7500, 3750, 5000, 6250, 7500, 10000, 10000]
    # Armar: torso (1 prismatic joint [0]) + one arm (8 revolute joints [1:9] from arm_cla_joint to arm_t8_joint)
    MAX_ACCELERATIONS_ARMAR = [15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]
    MAX_JERK_ARMAR = [7500, 7500, 7500, 7500, 7500, 7500, 7500, 7500, 7500]

    def __init__(self,
                 simulation_client_id=None,
                 simulation_time_step=None,
                 obstacle_client_id=None,
                 trajectory_time_step=None,
                 use_real_robot=False,
                 robot_scene=0,
                 obstacle_scene=0,
                 activate_obstacle_collisions=False,
                 observed_link_point_scene=0,
                 log_obstacle_data=False,
                 visual_mode=False,
                 visualize_bounding_spheres=False,
                 acc_range_function=None,
                 acc_braking_function=None,
                 collision_check_time=None,
                 check_braking_trajectory_collisions=False,
                 check_braking_trajectory_observed_points=False,
                 check_braking_trajectory_closest_points=True,
                 check_braking_trajectory_torque_limits=False,
                 closest_point_safety_distance=0.1,
                 observed_point_safety_distance=0.1,
                 use_target_points=False,
                 target_point_cartesian_range_scene=0,
                 target_point_relative_pos_scene=0,
                 target_point_radius=0.05,
                 target_point_sequence=0,
                 target_point_reached_reward_bonus=0.0,
                 target_point_use_actual_position=False,
                 no_self_collision=False,
                 target_link_name=None,
                 target_link_offset=None,
                 pos_limit_factor=1,
                 vel_limit_factor=1,
                 acc_limit_factor=1,
                 jerk_limit_factor=1,
                 torque_limit_factor=1,
                 **kwargs):

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

        if robot_scene <= 2:
            plane_z_offset = -0.94
            self._robot_name = "iiwa7"

        if robot_scene == 3 or robot_scene == 4:
            self._num_robots = 2
            plane_z_offset = 0
            self._robot_name = "armar6"
            if robot_scene == 3:
                robot_urdf = "armar6_front"
            else:
                robot_urdf = "armar6"

        if robot_scene == 5:
            self._num_robots = 4
            plane_z_offset = 0
            self._robot_name = "armar6_x4"
            robot_urdf = "armar6_x4_front"

        if target_link_name is None:
            if self._robot_name == "iiwa7":
                target_link_name = "iiwa_link_7"
            elif self._robot_name.startswith("armar6"):
                target_link_name = "hand_fixed"

        if target_link_offset is None:
            if self._robot_name == "iiwa7":
                target_link_offset = [0, 0, 0.126]
            elif self._robot_name.startswith("armar6"):
                target_link_offset = [0.03, 0, 0.135]
            else:
                target_link_offset = [0, 0, 0]

        if self._robot_name.startswith("armar6"):
            try:
                import armar
                urdf_path = armar.get_path_to_urdf_file(robot_name=robot_urdf)
            except ModuleNotFoundError:
                raise ValueError("Support for ARMAR robots requires the armar module. "
                                 "Run 'pip install armar' to install the module.")
        else:
            urdf_path = os.path.join(self.URDF_DIR, robot_urdf + ".urdf")

        for i in range(self._num_clients):
            if self._no_self_collision:
                self._robot_id = p.loadURDF(urdf_path, useFixedBase=True,
                                            physicsClientId=i)
            else:
                self._robot_id = p.loadURDF(urdf_path, useFixedBase=True,
                                            flags=p.URDF_USE_SELF_COLLISION,
                                            physicsClientId=i)

        for i in range(self._num_clients):
            half_extents = [10, 15, 0.0025]
            shape_visual = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=half_extents,
                                               rgbaColor=(0.9, 0.9, 0.9, 1), physicsClientId=i)
            shape_collision = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=half_extents,
                                                     physicsClientId=i)

            self._plane_id = p.createMultiBody(baseMass=0, basePosition=[0, 0, plane_z_offset],
                                               baseOrientation=(0.0, 0.0, 0.0, 1.0),
                                               baseVisualShapeIndex=shape_visual,
                                               baseCollisionShapeIndex=shape_collision,
                                               physicsClientId=i)

        joint_lower_limits, joint_upper_limits, force_limits, velocity_limits = [], [], [], []

        self._manip_joint_indices, self._manip_joint_indices_per_robot = self._get_manip_joint_indices()
        self._num_manip_joints = len(self._manip_joint_indices)
        self._link_name_list = []

        for i in range(p.getNumJoints(self._robot_id)):
            joint_info = p.getJointInfo(self._robot_id, i)
            self._link_name_list.append(
                str(joint_info[12])[2:-1])  # link name is loaded as  b'linkname' -> extract linkname

        for i in self._manip_joint_indices:
            joint_infos = p.getJointInfo(self._robot_id, i)
            if self._robot_name == "iiwa7":
                joint_limits_safety_buffer = self.JOINT_LIMITS_SAFETY_BUFFER_IIWA
            elif self._robot_name.startswith("armar6"):
                joint_limits_safety_buffer = self.JOINT_LIMITS_SAFETY_BUFFER_ARMAR
            else:
                joint_limits_safety_buffer = 0
            if joint_infos[8] == 0 and joint_infos[9] == -1.0:
                # continuous joint
                joint_lower_limits.append(np.nan)
                joint_upper_limits.append(np.nan)
            else:
                joint_lower_limits.append(joint_infos[8] + joint_limits_safety_buffer)
                joint_upper_limits.append(joint_infos[9] - joint_limits_safety_buffer)
            force_limits.append(joint_infos[10])
            velocity_limits.append(joint_infos[11])
        self._initial_joint_lower_limits = np.array(joint_lower_limits)
        self._initial_joint_upper_limits = np.array(joint_upper_limits)
        self._initial_max_torques = np.array(force_limits)
        self._initial_max_velocities = np.array(velocity_limits)
        if self._robot_name == "iiwa7":
            self._initial_max_accelerations = np.array(self.MAX_ACCELERATIONS_IIWA * self._num_robots)
            self._initial_max_jerk = np.array(self.MAX_JERK_IIWA * self._num_robots)
        if self._robot_name.startswith("armar6"):
            self._initial_max_accelerations = np.array(self.MAX_ACCELERATIONS_ARMAR[0:1] +
                                                       self.MAX_ACCELERATIONS_ARMAR[1:9] * self._num_robots)
            self._initial_max_jerk = np.array(self.MAX_JERK_ARMAR[0:1] +
                                              self.MAX_JERK_ARMAR[1:9] * self._num_robots)

        self._deactivate_self_collision_for_adjoining_links()
        self._obstacle_wrapper = \
            ObstacleWrapperSim(robot_scene=self,
                               simulation_client_id=self._simulation_client_id,
                               simulation_time_step=self._simulation_time_step,
                               obstacle_client_id=self._obstacle_client_id,
                               use_real_robot=self._use_real_robot,
                               visual_mode=visual_mode,
                               obstacle_scene=obstacle_scene,
                               activate_obstacle_collisions=activate_obstacle_collisions,
                               observed_link_point_scene=observed_link_point_scene,
                               log_obstacle_data=log_obstacle_data,
                               link_name_list=self._link_name_list,
                               manip_joint_indices=self._manip_joint_indices,
                               acc_range_function=acc_range_function,
                               acc_braking_function=acc_braking_function,
                               check_braking_trajectory_collisions=check_braking_trajectory_collisions,
                               check_braking_trajectory_torque_limits=check_braking_trajectory_torque_limits,
                               collision_check_time=collision_check_time,
                               check_braking_trajectory_observed_points=check_braking_trajectory_observed_points,
                               check_braking_trajectory_closest_points=check_braking_trajectory_closest_points,
                               closest_point_safety_distance=closest_point_safety_distance,
                               observed_point_safety_distance=observed_point_safety_distance,
                               use_target_points=use_target_points,
                               target_point_cartesian_range_scene=target_point_cartesian_range_scene,
                               target_point_relative_pos_scene=target_point_relative_pos_scene,
                               target_point_radius=target_point_radius,
                               target_point_sequence=target_point_sequence,
                               target_point_reached_reward_bonus=target_point_reached_reward_bonus,
                               target_point_use_actual_position=target_point_use_actual_position,
                               target_link_name=target_link_name,
                               target_link_offset=target_link_offset,
                               visualize_bounding_spheres=visualize_bounding_spheres
                               )

        self._joint_lower_limits = None
        self._joint_lower_limits_continuous = None
        self._joint_upper_limits = None
        self._joint_upper_limits_continuous = None
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
        self._joint_lower_limits_continuous = np.where(np.isnan(self._joint_lower_limits), -np.pi,
                                                       self._joint_lower_limits)
        self._joint_upper_limits = list(np.array(self._initial_joint_upper_limits) * self._pos_limit_factor)
        self._joint_upper_limits_continuous = np.where(np.isnan(self._joint_upper_limits), np.pi,
                                                       self._joint_upper_limits)
        self._max_velocities = self._initial_max_velocities * self._vel_limit_factor
        self._max_accelerations = self._initial_max_accelerations * self._acc_limit_factor
        self._max_jerk_linear_interpolation = np.array([min(2 * self._max_accelerations[i] / self._trajectory_time_step,
                                                            self._initial_max_jerk[i]) * self._jerk_limit_factor
                                                        for i in range(len(self._max_accelerations))])
        self._max_torques = self._initial_max_torques * self._torque_limit_factor

        if self._obstacle_wrapper is not None:
            self._obstacle_wrapper.trajectory_time_step = self._trajectory_time_step
            self._obstacle_wrapper.torque_limits = np.array([-1 * self._max_torques, self._max_torques])

        logging.info("Pos upper limits: %s", np.array(self._joint_upper_limits))
        logging.info("Pos lower limits: %s", np.array(self._joint_lower_limits))
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
    def joint_lower_limits_continuous(self):
        return self._joint_lower_limits_continuous

    @property
    def joint_upper_limits(self):
        return self._joint_upper_limits

    @property
    def joint_upper_limits_continuous(self):
        return self._joint_upper_limits_continuous

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
    def robot_name(self):
        return self._robot_name


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
                    if self._robot_name.startswith("armar6") and link_name == "torso":
                        joint_indices_per_robot[0].append(i)
                        joint_indices_per_robot[1].append(i)
                    else:
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

        joint_states = p.getJointStates(self._robot_id, manip_joint_indices, physicsClientId=physics_client_id)
        actual_torques = np.asarray([joint_state[3] for joint_state in joint_states])
        return actual_torques

    def _deactivate_self_collision_for_adjoining_links(self):
        # deactivate erroneous self-collisions resulting from inaccurate collision meshes
        if self._robot_name == "iiwa7":
            deactivate_self_collision_detection_link_name_pair_list = []
            deactivate_self_collision_detection_link_name_pair_list_per_robot = [["iiwa_link_5", "iiwa_link_7"]]
        elif self._robot_name.startswith("armar6"):
            deactivate_self_collision_detection_link_name_pair_list = []
            deactivate_self_collision_detection_link_name_pair_list_per_robot = [["lower_neck", "arm_cla"],
                                                                                 ["lower_neck", "arm_t12"],
                                                                                 ["torso", "arm_t12"],
                                                                                 ["arm_t12", "arm_t34"],
                                                                                 ["arm_t67", "arm_t8"]]
            if self._robot_name == "armar6_x4":
                deactivate_self_collision_detection_link_name_pair_list.extend([["torso", "arm_cla_r2"],
                                                                                ["torso", "arm_cla_r3"],
                                                                                ["platform", "arm_t12_r2"],
                                                                                ["platform", "arm_t12_r3"]])
        else:
            deactivate_self_collision_detection_link_name_pair_list = []
            deactivate_self_collision_detection_link_name_pair_list_per_robot = []

        for link_name_pair in deactivate_self_collision_detection_link_name_pair_list:
            self._deactivate_self_collision_detection(link_name_a=link_name_pair[0],
                                                      link_name_b=link_name_pair[1])

        for link_name_pair in deactivate_self_collision_detection_link_name_pair_list_per_robot:
            for j in range(self.num_robots):
                link_name_pair_robot = self.get_link_names_for_multiple_robots(link_name_pair, robot_indices=[j])
                self._deactivate_self_collision_detection(link_name_a=link_name_pair_robot[0],
                                                          link_name_b=link_name_pair_robot[1])

    def _deactivate_self_collision_detection(self, link_name_a, link_name_b):
        link_index_a = self.get_link_index_from_link_name(link_name_a)
        link_index_b = self.get_link_index_from_link_name(link_name_b)
        for j in range(self.num_clients):
            p.setCollisionFilterPair(self.robot_id, self.robot_id, link_index_a,
                                     link_index_b, enableCollision=0, physicsClientId=j)

    def get_link_names_for_multiple_robots(self, link_names, robot_indices=None):
        if isinstance(link_names, str):
            link_names = [link_names]

        if self._num_robots == 1:
            return link_names  # do nothing if there is one robot only

        link_names_multiple_robots = []

        if robot_indices is None:
            robot_indices = np.arange(self._num_robots)

        if self._robot_name.startswith("armar6"):
            shared_link_names = ['platform', 'torso', 'lower_neck', 'middle_neck', 'upper_neck']
        else:
            shared_link_names = []

        for i in range(len(robot_indices)):
            for j in range(len(link_names)):
                if link_names[j] in shared_link_names:
                    link_names_multiple_robots.append(link_names[j])
                else:
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
        pass

    @staticmethod
    def send_command_to_trajectory_controller(target_positions, **kwargs):
        raise NotImplementedError()
