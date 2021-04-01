# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import os
import pybullet as p
import datetime
from abc import ABC
from gym.wrappers.monitoring.video_recorder import VideoRecorder
from safemotions.envs.safe_motions_base import SafeMotionsBase

RENDER_MODES = ["human", "rgb_array"]
VIDEO_FRAME_RATE = 60
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080


def compute_view_matrix(camera_angle=0):
    if camera_angle == 0:
        cam_target_pos = (0, 0, 0)
        cam_dist = 1.75
        yaw = 90
        pitch = -70
        roll = 0

    elif camera_angle == 1:
        cam_target_pos = (-0.25, 0, 0)
        cam_dist = 1.95
        yaw = 90
        pitch = -40
        roll = 0

    elif camera_angle == 2:
        yaw = 59.59992599487305
        pitch = -49.400054931640625
        cam_dist = 2.000002861022949
        cam_target_pos = (0.0, 0.0, 0.0)
        roll = 0

    elif camera_angle == 3:
        yaw = 64.39994049072266
        pitch = -37.000003814697266
        cam_dist = 2.000002861022949
        cam_target_pos = (0.0, 0.0, 0.0)
        roll = 0

    elif camera_angle == 4:
        yaw = 69.59991455078125
        pitch = -33.8000602722168
        cam_dist = 1.8000028133392334
        cam_target_pos = (0.0, 0.0, 0.0)
        roll = 0

    elif camera_angle == 5:
        yaw = 90.800048828125
        pitch = -59.800079345703125
        cam_dist = 1.8000028133392334
        cam_target_pos = (0.0, 0.0, 0.0)
        roll = 0

    elif camera_angle == 6:
        yaw = 90.4000473022461
        pitch = -65.40008544921875
        cam_dist = 2.000002861022949
        cam_target_pos = (0.0, 0.0, 0.0)
        roll = 0

    elif camera_angle == 7:
        yaw = 90.00004577636719
        pitch = -45.4000358581543
        cam_dist = 2.000002861022949
        cam_target_pos = (0.0, 0.0, 0.0)
        roll = 0

    elif camera_angle == 8:
        yaw = 89.60002899169922
        pitch = -17.400007247924805
        cam_dist = 1.4000000953674316
        cam_target_pos = (-0.07712450623512268, 0.05323473736643791, 0.45070940256118774)
        roll = 0

    return p.computeViewMatrixFromYawPitchRoll(cam_target_pos, cam_dist, yaw, pitch, roll, 2)


def compute_projection_matrix():
    fov = 90
    aspect_ratio = VIDEO_WIDTH / VIDEO_HEIGHT
    near_distance = 0.1
    far_distance = 100

    return p.computeProjectionMatrixFOV(fov, aspect_ratio, near_distance, far_distance)


class VideoRecordingManager(ABC, SafeMotionsBase):
    def __init__(self,
                 experiment_name,
                 *vargs,
                 render_video=False,
                 extra_render_modes=None,
                 video_frame_rate=VIDEO_FRAME_RATE,
                 camera_angle=0,
                 **kwargs):
        super().__init__(*vargs, **kwargs)

        # video recording settings
        self._video_recorder = None
        self._render_video = render_video
        time_stamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        self._video_dir = os.path.join(self._evaluation_dir, self.__class__.__name__, experiment_name, time_stamp)
        self._video_base_path = None
        self._render_modes = RENDER_MODES.copy()
        if extra_render_modes:
            self._render_modes += extra_render_modes
        self._video_frame_rate = video_frame_rate
        self._sim_steps_per_frame = int(1 / (self._video_frame_rate * self._control_time_step))
        self._video_height = VIDEO_HEIGHT
        self._video_width = VIDEO_WIDTH
        self._camera_angle = camera_angle
        self._view_matrix = compute_view_matrix(self._camera_angle)
        self._projection_matrix = compute_projection_matrix()
        self._sim_step_counter = None

    @property
    def metadata(self):
        metadata = {
            'render.modes': self._render_modes,
            'video.frames_per_second': self._video_frame_rate
        }

        return metadata

    def reset(self):
        observation = super().reset()

        self._sim_step_counter = 0

        if self._render_video:
            self._reset_video_recorder()

        return observation

    def close(self):
        if self._video_recorder:
            self._close_video_recorder()

        super().close()

    def _sim_step(self):
        super()._sim_step()
        self._sim_step_counter += 1

        if self._render_video:
            if self._sim_step_counter == self._sim_steps_per_frame:
                self._capture_frame_with_video_recorder()

    def _prepare_for_end_of_episode(self):
        super()._prepare_for_end_of_episode()

        if self._render_video:
            for _ in range(self._video_frame_rate):
                self._capture_frame_with_video_recorder()
            if self._video_recorder:
                self._close_video_recorder()

    def _capture_frame_with_video_recorder(self):
        self._sim_step_counter = 0
        self._video_recorder.capture_frame()

    def _reset_video_recorder(self):
        if self._video_recorder:
            self._close_video_recorder()

        os.makedirs(self._video_dir, exist_ok=True)

        episode_id = self._episode_counter
        self._video_base_path = os.path.join(self._video_dir, "episode_{}".format(episode_id))
        metadata = {'episode_id': episode_id}
        self._video_recorder = VideoRecorder(self, base_path=self._video_base_path, metadata=metadata, enabled=True)

        for _ in range(self._video_frame_rate):
            self._capture_frame_with_video_recorder()

    def _close_video_recorder(self):
        self._video_recorder.close()
        self._video_recorder = []
