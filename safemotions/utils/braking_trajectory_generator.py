# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
from klimits.limit_calculation import PosVelJerkLimitation


def _denormalize(norm_value, value_range):
    actual_value = value_range[0] + 0.5 * (norm_value + 1) * (value_range[1] - value_range[0])
    return actual_value


class BrakingTrajectoryGenerator(object):

    def __init__(self,
                 trajectory_time_step,
                 acc_limits_break,
                 jerk_limits_break):

        self._trajectory_time_step = trajectory_time_step
        self._acc_limits_break = acc_limits_break
        self._jerk_limits_break = jerk_limits_break
        self._num_joints = len(self._acc_limits_break)
        self._vel_limits_break = [[0, 0]] * self._num_joints
        self._action_mapping_factor = 1.0

        self._acc_calculator = PosVelJerkLimitation(time_step=self._trajectory_time_step,
                                                    pos_limits=None, vel_limits=self._vel_limits_break,
                                                    acc_limits=self._acc_limits_break,
                                                    jerk_limits=self._jerk_limits_break,
                                                    acceleration_after_max_vel_limit_factor=0.0001,
                                                    set_velocity_after_max_pos_to_zero=True,
                                                    limit_velocity=True,
                                                    limit_position=False,
                                                    num_workers=1,
                                                    soft_velocity_limits=True,
                                                    soft_position_limits=False)

    def get_braking_acceleration(self, start_velocity, start_acceleration, index=0):
        if np.all(np.abs(start_velocity) < 0.01) and np.all(np.abs(start_acceleration) < 0.01):
            end_acceleration = [0.0] * self._num_joints
            robot_stopped = True
        else:
            robot_stopped = False

            norm_acc_range, _ = self._acc_calculator.calculate_valid_acceleration_range(current_pos=None,
                                                                                        current_vel=start_velocity,
                                                                                        current_acc=start_acceleration,
                                                                                        braking_trajectory=True,
                                                                                        time_step_counter=index)
            mapping_factor = []
            for j in range(self._num_joints):
                if start_velocity[j] < 0:
                    mapping_factor.append(self._action_mapping_factor)
                else:
                    mapping_factor.append(1 - self._action_mapping_factor)

            norm_end_acceleration = np.array([norm_acc_range[j][0] + mapping_factor[j] *
                                              (norm_acc_range[j][1] - norm_acc_range[j][0]) for j in
                                              range(len(mapping_factor))])

            norm_end_acceleration = [0.0 if abs(start_velocity[i]) < 0.01 and abs(start_acceleration[i] < 0.01) else
                                     norm_end_acceleration[i] for i in range(len(norm_end_acceleration))]

            end_acceleration = [_denormalize(norm_end_acceleration[k], self._acc_limits_break[k])
                                for k in range(len(norm_end_acceleration))]  # calculate actual acceleration

        return end_acceleration, robot_stopped

    def get_clipped_braking_acceleration(self, start_velocity, start_acceleration, next_acc_min, next_acc_max, index=0):
        end_acceleration, robot_stopped = self.get_braking_acceleration(start_velocity, start_acceleration, index)
        # avoid oscillations around p_max or p_min by reducing the valid range by x percent
        action_mapping_factor = 0.98
        next_acc_min = np.array(next_acc_min)
        next_acc_max = np.array(next_acc_max)
        next_acc_max_no_oscillation = next_acc_min + action_mapping_factor * (next_acc_max - next_acc_min)
        next_acc_min_no_oscillation = next_acc_min + (1 - action_mapping_factor) * (next_acc_max - next_acc_min)

        return np.clip(end_acceleration, next_acc_min_no_oscillation,
                       next_acc_max_no_oscillation), robot_stopped
