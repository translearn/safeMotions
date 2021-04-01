# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np


def clip_index(index, list_length):
    if index < 0 and abs(index) > list_length:
        return 0
    if index > 0 and index > list_length - 1:
        return -1
    else:
        return index


class TrajectoryManager(object):

    def __init__(self,
                 trajectory_time_step,
                 trajectory_duration,
                 obstacle_wrapper,
                 **kwargs):

        self._trajectory_time_step = trajectory_time_step
        self._trajectory_duration = trajectory_duration
        self._obstacle_wrapper = obstacle_wrapper

        self._trajectory_start_position = None
        self._trajectory_length = None
        self._num_manip_joints = None
        self._zero_joint_vector = None
        self._generated_trajectory = None
        self._measured_actual_trajectory_control_points = None
        self._computed_actual_trajectory_control_points = None
        self._generated_trajectory_control_points = None

        self._controller_model_coefficient_a = None
        self._controller_model_coefficient_b = None

    @property
    def trajectory_time_step(self):
        return self._trajectory_time_step

    @property
    def trajectory_length(self):
        return self._trajectory_length

    def reset(self, get_new_trajectory=True):
        if get_new_trajectory:
            self._trajectory_start_position = self._get_new_trajectory_start_position()
        self._trajectory_length = int(self._trajectory_duration / self._trajectory_time_step) + 1
        self._num_manip_joints = len(self._trajectory_start_position)
        self._zero_joint_vector = [0.0] * self._num_manip_joints
        self._generated_trajectory = {'positions': [self.get_trajectory_start_position()],
                                      'velocities': [self._zero_joint_vector],
                                      'accelerations': [self._zero_joint_vector]}
        self._measured_actual_trajectory_control_points = {'positions': [self.get_trajectory_start_position()],
                                                           'velocities': [self._zero_joint_vector],
                                                           'accelerations': [self._zero_joint_vector]}
        self._computed_actual_trajectory_control_points = {'positions': [self.get_trajectory_start_position()],
                                                           'velocities': [self._zero_joint_vector],
                                                           'accelerations': [self._zero_joint_vector]}
        self._generated_trajectory_control_points = {'positions': [self.get_trajectory_start_position()],
                                                     'velocities': [self._zero_joint_vector],
                                                     'accelerations': [self._zero_joint_vector]}

    def get_trajectory_start_position(self):
        return self._trajectory_start_position

    def _denormalize(self, norm_value, value_range):
        actual_value = value_range[0] + 0.5 * (norm_value + 1) * (value_range[1] - value_range[0])
        return actual_value

    def get_generated_trajectory_point(self, index, key='positions'):
        i = clip_index(index, len(self._generated_trajectory[key]))

        return self._generated_trajectory[key][i]

    def get_measured_actual_trajectory_control_point(self, index, key='positions'):
        i = clip_index(index, len(self._measured_actual_trajectory_control_points[key]))

        return self._measured_actual_trajectory_control_points[key][i]

    def get_computed_actual_trajectory_control_point(self, index, key='positions'):
        i = clip_index(index, len(self._computed_actual_trajectory_control_points[key]))

        return self._computed_actual_trajectory_control_points[key][i]

    def get_generated_trajectory_control_point(self, index, key='positions'):
        i = clip_index(index, len(self._generated_trajectory_control_points[key]))

        return self._generated_trajectory_control_points[key][i]

    def add_generated_trajectory_point(self, positions, velocities, accelerations):
        self._generated_trajectory['positions'].append(positions)
        self._generated_trajectory['velocities'].append(velocities)
        self._generated_trajectory['accelerations'].append(accelerations)

    def add_measured_actual_trajectory_control_point(self, positions, velocities, accelerations):
        self._measured_actual_trajectory_control_points['positions'].append(positions)
        self._measured_actual_trajectory_control_points['velocities'].append(velocities)
        self._measured_actual_trajectory_control_points['accelerations'].append(accelerations)

    def add_computed_actual_trajectory_control_point(self, positions, velocities, accelerations):
        self._computed_actual_trajectory_control_points['positions'].append(positions)
        self._computed_actual_trajectory_control_points['velocities'].append(velocities)
        self._computed_actual_trajectory_control_points['accelerations'].append(accelerations)

    def add_generated_trajectory_control_point(self, positions, velocities, accelerations):
        self._generated_trajectory_control_points['positions'].append(positions)
        self._generated_trajectory_control_points['velocities'].append(velocities)
        self._generated_trajectory_control_points['accelerations'].append(accelerations)

    def compute_controller_model_coefficients(self, time_constants, sampling_time):
        self._controller_model_coefficient_a = 1 + (2 * np.array(time_constants) / sampling_time)
        self._controller_model_coefficient_b = 1 - (2 * np.array(time_constants) / sampling_time)

    def model_position_controller_to_compute_actual_position(self, current_position_setpoint, last_position_setpoint):
        # models the position controller as a discrete transfer function and returns the
        # computed actual position, given the next position setpoint and previous computed actual positions
        # the controller is modelled as a first order low-pass with a (continuous) transfer function of
        #  G(s) = 1 / (1 + T * s)
        # the transfer function is discretized using Tustins approximation: s = 2 / Ta * (z - 1) / (z + 1)
        # the following difference equation can be derived:
        # y_n = 1/a * (x_n + x_n_minus_one - b * y_n_minus_one) with a = 1 + (2 * T / Ta) and b = 1 - (2 * T / Ta)

        x_n = np.array(current_position_setpoint)
        x_n_minus_one = np.array(last_position_setpoint)
        y_n_minus_one = np.array(self.get_computed_actual_trajectory_control_point(-1))
        computed_actual_position = 1 / self._controller_model_coefficient_a * \
                                   (x_n + x_n_minus_one - self._controller_model_coefficient_b * y_n_minus_one)
        return computed_actual_position

    def is_trajectory_finished(self, index):
        return index >= self._trajectory_length - 1

    def _get_new_trajectory_start_position(self):
        return self._obstacle_wrapper.get_starting_point_joint_pos()



