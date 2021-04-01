# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import json
import numpy as np
import os
import inspect
import errno
import datetime
import pybullet as p
from collections import defaultdict
from glob import glob
from itertools import chain
from safemotions.envs.decorators import actions, observations, rewards, video_recording

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))


class SafeMotionsEnv(actions.AccelerationPredictionBoundedJerkAccVelPos,
                     observations.SafeObservation,
                     rewards.TargetPointReachingReward,
                     video_recording.VideoRecordingManager):
    def __init__(self,
                 experiment_name,
                 *vargs,
                 **kwargs):
        super().__init__(experiment_name, *vargs, **kwargs)
        self._experiment_name = experiment_name
        self._timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

    def render(self, mode="human"):
        if mode == "human":
            return np.array([])
        else:
            (_, _, image, _, _) = p.getCameraImage(width=self._video_width, height=self._video_height,
                                                   renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                                   viewMatrix=self._view_matrix,
                                                   projectionMatrix=self._projection_matrix)
            image = np.reshape(image, (self._video_height, self._video_width, 4))
            image = np.uint8(image[:, :, :3])

            return np.array(image)

    def _process_action_outcome(self, base_info, action_info):
        reward, reward_info = self._get_reward()
        observation, observation_info = self._get_observation()
        done = self._check_termination()
        info = defaultdict(dict)

        for k, v in chain(base_info.items(), action_info.items(), observation_info.items(), reward_info.items()):
            info[k] = {**info[k], **v}

        return observation, reward, done, info

    def _process_end_of_episode(self, observation, reward, done, info):
        info.update(trajectory_length=self._trajectory_manager.trajectory_length)
        info.update(episode_length=self._episode_length)
        info['termination_reason'] = self._termination_reason

        # get info from obstacle wrapper
        obstacle_info = self._robot_scene.obstacle_wrapper.get_info_and_print_stats()
        info = dict(info, **obstacle_info)  # concatenate dicts
        self._display_plot()
        self._save_plot(self.__class__.__name__, self._experiment_name)

        return observation, reward, done, info

    def _store_action_list(self):
        action_dict = {'actions': np.asarray(self._action_list).tolist()}
        eval_dir = os.path.join(project_dir, "evalution", "action_logs",
                                self.__class__.__name__, self._experiment_name, self._timestamp)

        if not os.path.exists(eval_dir):
            try:
                os.makedirs(eval_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        with open(os.path.join(eval_dir, "episode_" + str(self._episode_counter) + ".json"), 'w') as f:
            f.write(json.dumps(action_dict))
            f.flush()

    def _create_metadata(self):
        return {'episode_id': self._episode_counter, 'trajectory_length': self._trajectory_manager.trajectory_length,
                'episode_length': self._episode_length, 'total_reward': self._total_reward}

    def _close_video_recorder(self):
        super()._close_video_recorder()
        self._adapt_rendering_metadata()
        self._rename_output_files()

    def _adapt_rendering_metadata(self):
        metadata_ext = ".meta.json"
        metadata_file = self._video_base_path + metadata_ext

        with open(metadata_file, 'r') as f:
            metadata_json = json.load(f)

            encoder_metadata = metadata_json.pop('encoder_version', None)
            if encoder_metadata:
                metadata_json.update(encoder=encoder_metadata['backend'])

            metadata_json.update(trajectory_length=self._trajectory_manager.trajectory_length)
            metadata_json.update(episode_length=self._episode_length)
            metadata_json.update(total_reward=round(self._total_reward, 3))

        with open(metadata_file, 'w') as f:
            f.write(json.dumps(metadata_json, indent=4))
            f.close()

    def _rename_output_files(self):
        output_file = glob(self._video_base_path + ".*")

        for file in output_file:
            dir_path, file_name = os.path.split(file)
            name, extension = os.path.splitext(file_name)
            new_name = "_".join(map(str, [name, self._episode_length,
                                          self._robot_scene.obstacle_wrapper.get_num_target_points_reached(),
                                          round(self._total_reward, 3)]))
            new_file_name = new_name + extension
            os.rename(file, os.path.join(dir_path, new_file_name))
