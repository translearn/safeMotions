# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import json
import numpy as np
import os
import inspect
import errno
from collections import defaultdict
from itertools import chain
from safemotions.envs.decorators import actions, observations, rewards, video_recording

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))


class SafeMotionsEnv(actions.AccelerationPredictionBoundedJerkAccVelPos,
                     observations.SafeObservation,
                     rewards.TargetPointReachingReward,
                     video_recording.VideoRecordingManager):
    def __init__(self,
                 *vargs,
                 **kwargs):
        super().__init__(*vargs, **kwargs)

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
        eval_dir = os.path.join(self._evaluation_dir, "action_logs")

        if not os.path.exists(eval_dir):
            try:
                os.makedirs(eval_dir)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

        with open(os.path.join(eval_dir, "episode_{}_{}.json".format(self._episode_counter, self.pid)), 'w') as f:
            f.write(json.dumps(action_dict))
            f.flush()
