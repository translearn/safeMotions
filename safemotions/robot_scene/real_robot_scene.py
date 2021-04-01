# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from safemotions.robot_scene.robot_scene_base import RobotSceneBase
import pybullet as p
from abc import abstractmethod


class RealRobotScene(RobotSceneBase):

    def __init__(self,
                 *vargs,
                 real_robot_debug_mode=False,
                 **kwargs):

        super().__init__(**kwargs)

        self._real_robot_debug_mode = real_robot_debug_mode

        if not self._real_robot_debug_mode:
            print("Waiting for real robot ...")
            self._connect_to_real_robot()
            print("Connected to real robot")
        else:
            print("Real robot debug mode: Commands are not send to the real robot")

    @abstractmethod
    def _connect_to_real_robot(self):
        raise NotImplementedError()

    def pose_manipulator(self, joint_positions, **kwargs):

        if self._simulation_client_id is not None:
            # set the simulation client to the desired starting position
            for i in range(len(self._manip_joint_indices)):
                p.resetJointState(bodyUniqueId=self._robot_id,
                                  jointIndex=self._manip_joint_indices[i],
                                  targetValue=joint_positions[i],
                                  targetVelocity=0,
                                  physicsClientId=self._simulation_client_id)

        go_on = ""
        while go_on != "YES":
            go_on = input("Type in 'yes' to move to the starting position").upper()

        if not self._real_robot_debug_mode:
            success = self._move_to_joint_position(joint_positions)
        else:
            success = True

        return success

    @abstractmethod
    def _move_to_joint_position(self, joint_positions):
        raise NotImplementedError()

    def set_motor_control(self, target_positions, physics_client_id=None, computed_position_is=None,
                          computed_velocity_is=None, **kwargs):
        if physics_client_id is None:
            if not self._real_robot_debug_mode:
                self._send_command_to_trajectory_controller(target_positions, **kwargs)
            if self._simulation_client_id is not None and \
                    computed_position_is is not None \
                    and computed_velocity_is is not None:
                for i in range(len(self._manip_joint_indices)):
                    p.resetJointState(bodyUniqueId=self._robot_id,
                                      jointIndex=self._manip_joint_indices[i],
                                      targetValue=computed_position_is[i],
                                      targetVelocity=computed_velocity_is[i],
                                      physicsClientId=self._simulation_client_id)
        else:
            super().set_motor_control(target_positions, physics_client_id=physics_client_id, **kwargs)

    @abstractmethod
    def _send_command_to_trajectory_controller(self, target_positions, **kwargs):
        raise NotImplementedError()

    @abstractmethod
    def _read_actual_torques(self):
        raise NotImplementedError()

    def get_actual_joint_torques(self, physics_client_id=None, **kwargs):
        if physics_client_id is None:
            return self._read_actual_torques()
        else:
            return super().get_actual_joint_torques(physics_client_id=physics_client_id, **kwargs)

    @abstractmethod
    def get_actual_joint_position_and_velocity(self, manip_joint_indices):
        raise NotImplementedError()

    def disconnect(self):
        super().disconnect()

    def prepare_for_end_of_episode(self):
        pass

    def prepare_for_start_of_episode(self):
        input("Press key to start the online trajectory generation")


