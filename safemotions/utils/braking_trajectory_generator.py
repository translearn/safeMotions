# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
from klimits import PosVelJerkLimitation
from klimits import denormalize


def _denormalize(norm_value, value_range):
    actual_value = value_range[0] + 0.5 * (norm_value + 1) * (value_range[1] - value_range[0])
    return actual_value


class BrakingTrajectoryGenerator(object):

    def __init__(self,
                 trajectory_time_step,
                 acc_limits_braking,
                 jerk_limits_braking):

        self._trajectory_time_step = trajectory_time_step
        self._acc_limits_braking = acc_limits_braking
        self._acc_limits_braking_min_max = np.swapaxes(self._acc_limits_braking, 0, 1)
        self._jerk_limits_braking = jerk_limits_braking
        self._num_joints = len(self._acc_limits_braking)
        self._vel_limits_braking = [[0, 0]] * self._num_joints
        self._action_mapping_factor = 1.0

        self._acc_calculator = PosVelJerkLimitation(time_step=self._trajectory_time_step,
                                                    pos_limits=None, vel_limits=self._vel_limits_braking,
                                                    acc_limits=self._acc_limits_braking,
                                                    jerk_limits=self._jerk_limits_braking,
                                                    acceleration_after_max_vel_limit_factor=0.0000,
                                                    set_velocity_after_max_pos_to_zero=True,
                                                    limit_velocity=True,
                                                    limit_position=False,
                                                    num_threads=None,
                                                    soft_velocity_limits=True,
                                                    soft_position_limits=False,
                                                    normalize_acc_range=False)

    def get_braking_acceleration(self, start_velocity, start_acceleration, index=0):
        if np.all(np.abs(start_velocity) < 0.01) and np.all(np.abs(start_acceleration) < 0.01):
            end_acceleration = np.zeros(self._num_joints)
            robot_stopped = True
        else:
            robot_stopped = False

            if self._action_mapping_factor == 1:
                limit_min_max = np.array([(0.0, 1.0) if start_velocity[i] < 0 else (1.0, 0.0)
                                          for i in range(len(start_velocity))])
            else:
                limit_min_max = None

            safe_acc_range, _ = self._acc_calculator.calculate_valid_acceleration_range(current_pos=None,
                                                                                        current_vel=start_velocity,
                                                                                        current_acc=start_acceleration,
                                                                                        braking_trajectory=True,
                                                                                        time_step_counter=index,
                                                                                        limit_min_max=limit_min_max)

            safe_acc_range_min_max = safe_acc_range.T
            if self._action_mapping_factor == 1:
                end_acceleration = np.where(start_velocity < 0, safe_acc_range_min_max[1],
                                            safe_acc_range_min_max[0])
            else:
                normalized_mapping_factor = np.where(start_velocity < 0, self._action_mapping_factor * 2 - 1,
                                                     (1 - self._action_mapping_factor) * 2 - 1)

                end_acceleration = denormalize(normalized_mapping_factor, safe_acc_range_min_max)

            end_acceleration = np.where(np.logical_and(np.abs(start_velocity) < 0.01,
                                                       np.abs(start_acceleration) < 0.01),
                                        0.0, end_acceleration)

        return end_acceleration, robot_stopped

    def get_clipped_braking_acceleration(self, start_velocity, start_acceleration, next_acc_min, next_acc_max, index=0):
        end_acceleration, robot_stopped = self.get_braking_acceleration(start_velocity, start_acceleration, index)
        # avoid oscillations around p_max or p_min by reducing the valid range by x percent
        action_mapping_factor = 1.0

        if action_mapping_factor != 1.0:
            next_acc_diff = next_acc_max - next_acc_min
            next_acc_max_no_oscillation = next_acc_min + action_mapping_factor * next_acc_diff
            next_acc_min_no_oscillation = next_acc_min + (1 - action_mapping_factor) * next_acc_diff
        else:
            next_acc_max_no_oscillation = next_acc_max
            next_acc_min_no_oscillation = next_acc_min

        return np.core.umath.clip(end_acceleration, next_acc_min_no_oscillation, next_acc_max_no_oscillation), robot_stopped
        
