#!/usr/bin/env python

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import json
import os
import sys
import inspect
import ray
import klimits
import datetime
import time
import logging
import numpy as np
from ray.rllib import rollout
from ray import tune
from pathlib import Path
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import AgentID, PolicyID
from typing import Dict

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))

# Termination reason
TERMINATION_UNSET = -1
TERMINATION_SUCCESS = 0
TERMINATION_JOINT_LIMITS = 1
TERMINATION_TRAJECTORY_LENGTH = 2

RENDERER = {'opengl': 0,
            'egl': 1,
            'cpu': 2}


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def get_metrics_dir(base_dir, real_robot):
    if real_robot:
        metrics_dir = os.path.join(base_dir, "trajectory_logs_real")
    else:
        metrics_dir = os.path.join(base_dir, "trajectory_logs_sim")

    return metrics_dir


def make_metrics_dir(base_dir, real_robot):
    metrics_dir = get_metrics_dir(base_dir, real_robot)
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), default=np_encoder))
        f.flush()


def get_network_data_dir(base_dir, real_robot):
    if real_robot:
        metrics_dir = os.path.join(base_dir, "network_data_real")
    else:
        metrics_dir = os.path.join(base_dir, "network_data_sim")

    return metrics_dir


def make_network_data_dir(base_dir, real_robot):
    network_data_dir = get_network_data_dir(base_dir, real_robot)
    os.makedirs(network_data_dir, exist_ok=True)
    with open(os.path.join(network_data_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), default=np_encoder))
        f.flush()


def store_network_data(base_dir, real_robot, pid, episode_counter, reward_total, network_data_list):
    network_data_file = "episode_{}_{}_{:.3f}.json".format(episode_counter, pid, reward_total)
    network_data_dir = get_network_data_dir(base_dir, real_robot)

    for i in range(len(network_data_list)):
        for key, value in network_data_list[i].items():
            if isinstance(value, np.ndarray):
                network_data_list[i][key] = list(value)

    with open(os.path.join(network_data_dir, network_data_file), 'w') as f:
        f.write(json.dumps(network_data_list, default=np_encoder, sort_keys=True))
        f.flush()


def store_metrics(base_dir, real_robot, pid, episode_counter, reward_total, last_info, episode_info):
    metric_file = "episode_{}_{}_{:.3f}.json".format(episode_counter, pid, reward_total)
    episode_info['reward'] = float(reward_total)
    episode_info['episode_length'] = int(last_info['episode_length'])
    episode_info['trajectory_length'] = int(last_info['trajectory_length'])
    for key, value in last_info.items():
        if key.startswith("obstacles"):
            episode_info[key] = value

    for key, value in episode_info['max'].items():
        episode_info['max'][key] = float(np.max(np.array(value)))
    for key, value in episode_info['average'].items():
        episode_info['average'][key] = float(np.mean(np.array(value)))
    for key, value in episode_info['min'].items():
        episode_info['min'][key] = float(np.min(np.array(value)))

    if last_info['termination_reason'] == TERMINATION_SUCCESS:
        episode_info['success_rate'] = int(1.0)
    else:
        episode_info['success_rate'] = int(0.0)

    if last_info['termination_reason'] == TERMINATION_JOINT_LIMITS:
        episode_info['joint_limit_violation_termination_rate'] = int(1.0)
    else:
        episode_info['joint_limit_violation_termination_rate'] = int(0.0)

    metrics_dir = get_metrics_dir(base_dir, real_robot)
    with open(os.path.join(metrics_dir, metric_file), 'w') as f:
        f.write(json.dumps(episode_info, default=np_encoder))
        f.flush()


def rollout_multiple_workers():
    remote_workers = agent.workers.remote_workers()
    if args.store_metrics or args.store_network_data:
        evaluation_dir = ray.get(remote_workers[0].foreach_env.remote(lambda env: env.evaluation_dir))[0]
        if args.store_metrics:
            make_metrics_dir(evaluation_dir, args.use_real_robot)
        if args.store_network_data:
            make_network_data_dir(evaluation_dir, args.use_real_robot)
    if args.seed is not None:  # increment the seed for each worker by one
        for i in range(0, len(remote_workers)):
            remote_workers[i].foreach_env.remote(lambda env: env.set_seed(args.seed + i))
    for _ in range(args.episodes):  # episodes per worker
        ray.get([worker.sample.remote() for worker in remote_workers])

    ray.get([worker.stop.remote() for worker in remote_workers])


def rollout_single_worker_manually():
    episodes_sampled = 0
    episode_computation_time_list = []
    start = time.time()

    if args.seed is not None:
        env.set_seed(args.seed)

    while True:
        if args.episodes:
            if episodes_sampled >= args.episodes:
                break

        network_data_list = []
        obs = env.reset()
        done = False
        reward_total = 0.0
        episode_info = {}
        steps = -1
        start_episode_timer = time.time()
        while not done:
            steps = steps + 1
            if args.store_network_data:
                network_data = agent.compute_action(obs, full_fetch=True)
                action = network_data[0]
                network_data[2]['action'] = action  # add action to extra_outs
                network_data[2]['observation'] = obs
                network_data_list.append(network_data[2])
            else:
                action = agent.compute_action(obs, full_fetch=False)
            if args.store_metrics:
                next_obs, reward, done, info = env.step(action)
                if episode_info:
                    for key, value in episode_info.items():
                        for key_2, value_2 in episode_info[key].items():
                            episode_info[key][key_2].append(info[key][key_2])
                else:
                    for key, value in info.items():
                        episode_info[key] = {}
                        if isinstance(info[key], dict):
                            for key_2, value_2 in info[key].items():
                                episode_info[key][key_2] = [info[key][key_2]]
            else:
                next_obs, reward, done, _ = env.step(action)

            reward_total += reward
            obs = next_obs

        end_episode_timer = time.time()
        episode_computation_time = end_episode_timer - start_episode_timer
        logging.info("Computing episode %s took %s seconds", episodes_sampled + 1, episode_computation_time)
        episode_computation_time_list.append(episode_computation_time)
        if env.precomputation_time is not None:
            logging.info("Trajectory duration: %s seconds. Control phase: %s seconds.",
                         (steps + 1) * env.trajectory_time_step, episode_computation_time - env.precomputation_time)
        else:
            logging.info("Trajectory duration: %s seconds", (steps + 1) * env.trajectory_time_step)
        logging.info("Episode reward: %s", reward_total)
        episodes_sampled += 1

        if args.store_metrics:
            store_metrics(env.evaluation_dir, env.use_real_robot, env.pid, episodes_sampled, reward_total,
                          info, episode_info)
        if args.store_network_data:
            store_network_data(env.evaluation_dir, env.use_real_robot, env.pid, episodes_sampled, reward_total,
                               network_data_list)

    end = time.time()
    env.close()
    logging.info("Computed %s episode(s) in %s seconds.", len(episode_computation_time_list), end - start)
    logging.info("Mean computation time: %s seconds, Max computation time: %s seconds.",
                 np.mean(episode_computation_time_list),
                 np.max(episode_computation_time_list))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None,
                        help="The name of the evaluation.")
    parser.add_argument('--evaluation_dir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the checkpoint for evaluation.")
    parser.add_argument('--episodes', type=int, default=20,
                        help="The number of episodes for evaluation per worker.")
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_threads_per_worker', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--use_real_robot', action='store_true', default=None)
    parser.add_argument('--real_robot_debug_mode', dest='real_robot_debug_mode', action='store_true', default=False)
    parser.add_argument('--use_gui', action='store_true', default=False)
    parser.add_argument('--online_trajectory_duration', type=float, default=None)
    parser.add_argument('--collision_check_time', type=float, default=None)
    parser.add_argument('--store_metrics', action='store_true', default=False)
    parser.add_argument('--plot_trajectory', action='store_true', default=False)
    parser.add_argument('--save_trajectory_plot', action='store_true', default=False)
    parser.add_argument('--plot_acc_limits', action='store_true', default=False)
    parser.add_argument('--plot_actual_values', action='store_true', default=False)
    parser.add_argument('--plot_actual_torques', action='store_true', default=False)
    parser.add_argument('--plot_computed_actual_values', action='store_true', default=False)
    parser.add_argument('--plot_joint', type=json.loads, default=None)
    parser.add_argument('--store_actions', action='store_true', default=False)
    parser.add_argument('--store_network_data', action='store_true', default=False)
    parser.add_argument('--no_exploration', action='store_true', default=False)
    parser.add_argument('--log_obstacle_data', action='store_true', default=False)
    parser.add_argument('--obstacle_scene', type=int, default=None)
    parser.add_argument('--obstacle_use_computed_actual_values', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--solver_iterations', type=int, default=None)
    parser.add_argument('--random_agent', action='store_true', default=False)
    parser.add_argument('--no_self_collision', action='store_true', default=False)
    parser.add_argument('--use_thread_for_movement', action='store_true', default=False)
    parser.add_argument('--use_process_for_movement', action='store_true', default=False)
    parser.add_argument('--control_time_step', type=float, default=None)
    parser.add_argument('--time_step_fraction_sleep_observation', type=float, default=None)
    parser.add_argument("--logging_level", default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--render', action='store_true', default=False,
                        help="If set, videos of the generated episodes are recorded.")
    parser.add_argument("--renderer", default='opengl', choices=['opengl', 'egl', 'cpu'])
    parser.add_argument('--camera_angle', type=int, default=0)
    parser.add_argument('--video_frame_rate', type=float, default=None)
    parser.add_argument('--video_height', type=int, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--video_add_text', action='store_true', default=False)
    parser.add_argument('--fixed_video_filename', action='store_true', default=False)
    parser.add_argument('--use_dashboard', action='store_true', default=False)

    args = parser.parse_args()

    if args.render and args.renderer == 'egl':
        os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
        os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

    if args.evaluation_dir is None:
        evaluation_dir = os.path.join(Path.home(), "safe_motions_evaluation")
    else:
        evaluation_dir = os.path.join(args.evaluation_dir, "safe_motions_evaluation")

    if not os.path.isdir(args.checkpoint) and not os.path.isfile(args.checkpoint):
        checkpoint_path = os.path.join(current_dir, "trained_networks", args.checkpoint)
    else:
        checkpoint_path = args.checkpoint

    if os.path.isdir(checkpoint_path):
        if os.path.basename(checkpoint_path) == "checkpoint":
            checkpoint_path = os.path.join(checkpoint_path, "checkpoint")
        else:
            checkpoint_path = os.path.join(checkpoint_path, "checkpoint", "checkpoint")

    if not os.path.isfile(checkpoint_path):
        raise ValueError("Could not find checkpoint {}".format(checkpoint_path))

    params_dir = os.path.dirname(os.path.dirname(checkpoint_path))
    params_path = os.path.join(params_dir, "params.json")

    with open(params_path) as params_file:
        checkpoint_config = json.load(params_file)
    checkpoint_config['evaluation_interval'] = None
    env_config = checkpoint_config['env_config']

    if args.name is not None:
        env_config['experiment_name'] = args.name

    logging.basicConfig()
    logging.getLogger().setLevel(args.logging_level)
    env_config['logging_level'] = args.logging_level

    if args.render:
        env_config.update(render_video=True)
        env_config['camera_angle'] = args.camera_angle
        env_config['renderer'] = RENDERER[args.renderer]
        env_config['video_frame_rate'] = args.video_frame_rate
        env_config['video_height'] = args.video_height
        env_config['video_dir'] = args.video_dir
        env_config['fixed_video_filename'] = args.fixed_video_filename
        env_config['video_add_text'] = args.video_add_text
        if args.fixed_video_filename and args.num_workers is not None and args.num_workers >= 2:
            raise ValueError("fixed_video_filename requires num_workers < 2")

    else:
        env_config.update(render_video=False)

    if args.use_gui:
        env_config.update(use_gui=True)
    else:
        env_config.update(use_gui=False)

    env_config.update(use_real_robot=args.use_real_robot)
    env_config['time_stamp'] = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')

    if args.store_actions:
        env_config['store_actions'] = True

    if args.log_obstacle_data:
        env_config['log_obstacle_data'] = True

    if args.save_trajectory_plot:
        env_config['save_trajectory_plot'] = True

    if args.plot_actual_torques:
        env_config['plot_actual_torques'] = True

    if args.use_thread_for_movement:
        env_config['use_thread_for_movement'] = True

    if args.use_process_for_movement:
        env_config['use_process_for_movement'] = True

    if args.obstacle_use_computed_actual_values:
        env_config['obstacle_use_computed_actual_values'] = True

    if args.obstacle_scene is not None:
        env_config['obstacle_scene'] = args.obstacle_scene

    if args.online_trajectory_duration is not None:
        env_config['online_trajectory_duration'] = args.online_trajectory_duration

    if args.plot_trajectory:

        env_config['plot_trajectory'] = True

        if args.plot_acc_limits:
            env_config['plot_acc_limits'] = True

        if args.plot_actual_values:
            env_config['plot_actual_values'] = True

        if args.plot_computed_actual_values:
            env_config['plot_computed_actual_values'] = True

        if args.plot_joint is not None:
            env_config['plot_joint'] = args.plot_joint

    if args.random_agent:
        env_config['random_agent'] = True

    if args.collision_check_time is not None:
        env_config['collision_check_time'] = args.collision_check_time

    if args.solver_iterations:
        env_config['solver_iterations'] = args.solver_iterations

    if args.real_robot_debug_mode:
        env_config['real_robot_debug_mode'] = True

    if args.no_self_collision:
        env_config['no_self_collision'] = True

    if args.time_step_fraction_sleep_observation is not None:
        env_config['time_step_fraction_sleep_observation'] = args.time_step_fraction_sleep_observation

    if args.control_time_step is not None:
        env_config['control_time_step'] = args.control_time_step

    checkpoint_config['num_workers'] = args.num_workers

    if args.num_gpus is not None:
        checkpoint_config['num_gpus'] = args.num_gpus
        if args.num_gpus == 0:
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"

    if args.seed is not None:
        checkpoint_config['seed'] = args.seed
        env_config['seed'] = args.seed

    if 'ray_version' in env_config and hasattr(ray, '__version__'):
        if env_config['ray_version'] != ray.__version__:
            logging.warning('This network was trained with ray=={} but you are using ray=={}'.format(
                env_config['ray_version'], ray.__version__))

    if 'klimits_version' in env_config and hasattr(klimits, '__version__'):
        if env_config['klimits_version'] != klimits.__version__:
            logging.warning('This network was trained with klimits=={} but you are using klimits=={}'.format(
                env_config['klimits_version'], klimits.__version__))

    if 'use_braking_trajectory_method' in env_config:  # for compatibility with older versions
        env_config['check_braking_trajectory_collisions'] = env_config['use_braking_trajectory_method']
        del env_config['use_braking_trajectory_method']

    checkpoint_config['env_config'] = env_config

    if 'sample_batch_size' in checkpoint_config:
        del checkpoint_config['sample_batch_size']
    checkpoint_config['rollout_fragment_length'] = 1  # stop sampling of remote workers after a single episode

    if args.no_exploration:
        checkpoint_config['explore'] = False

    if 'custom_model' in checkpoint_config['model']:
        from ray.rllib.models import ModelCatalog
        if checkpoint_config['model']['custom_model'] == 'fcnet_last_layer_activation':
            from safemotions.model.fcnet_v2_last_layer_activation import FullyConnectedNetworkLastLayerActivation
            ModelCatalog.register_custom_model('fcnet_last_layer_activation', FullyConnectedNetworkLastLayerActivation)
        if checkpoint_config['model']['custom_model'] == 'keras_fcnet_last_layer_activation':
            from safemotions.model.keras_fcnet_last_layer_activation import FullyConnectedNetworkLastLayerActivation
            ModelCatalog.register_custom_model('keras_fcnet_last_layer_activation',
                                               FullyConnectedNetworkLastLayerActivation)
            for key in ['fcnet_hiddens', 'fcnet_activation', 'post_fcnet_hiddens', 'post_fcnet_activation',
                        'no_final_layer', 'vf_share_layers', 'free_log_std']:
                if key in checkpoint_config['model'] and key not in checkpoint_config['model']['custom_model_config']:
                    checkpoint_config['model']['custom_model_config'][key] = checkpoint_config['model'][key]
        if 'custom_options' in checkpoint_config['model']:
            checkpoint_config['model']['custom_model_config'] = checkpoint_config['model']['custom_options']
            del checkpoint_config['model']['custom_options']

        if args.store_network_data:
            checkpoint_config['model']['custom_model_config']['output_intermediate_layers'] = True

    args.config = checkpoint_config
    args.run = "PPO"
    args.env = checkpoint_config['env']
    args.out = None
    if args.seed is not None:
        np.random.seed(args.seed)

    # define number of threads per worker for parallel execution based on OpenMP
    os.environ['OMP_NUM_THREADS'] = str(args.num_threads_per_worker)

    from safemotions.envs.safe_motions_env import SafeMotionsEnv
    from safemotions.train import CustomTrainCallbacks

    class CustomEvaluationCallbacks(CustomTrainCallbacks):

        def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                             policies: Dict[str, Policy],
                             episode: MultiAgentEpisode, env_index: int, **kwargs):

            episode.user_data['start_time'] = time.time()
            if args.store_metrics:
                super().on_episode_start(worker=worker, base_env=base_env,
                                         policies=policies,
                                         episode=episode, env_index=env_index, **kwargs)

        def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                            episode: MultiAgentEpisode, env_index: int, **kwargs):
            if args.store_metrics:
                super().on_episode_step(worker=worker, base_env=base_env,
                                        episode=episode, env_index=env_index, **kwargs)

        def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                           policies: Dict[str, Policy], episode: MultiAgentEpisode,
                           env_index: int, **kwargs):
            episode_computation_time = time.time() - episode.user_data['start_time']
            env = base_env.get_unwrapped()[-1]
            print("Computing episode {} took {} seconds".format(env.episode_counter, episode_computation_time))
            last_info = episode.last_info_for()
            episode_length = last_info['episode_length']
            print("Trajectory duration: {} seconds".format(episode_length * env.trajectory_time_step))
            reward_total = episode.agent_rewards[('agent0', 'default_policy')]
            print("Episode reward: {}".format(reward_total))
            if args.store_metrics:
                store_metrics(env.evaluation_dir, env.use_real_robot, env.pid,
                              env.episode_counter, reward_total, last_info, episode.user_data['op'])
            if 'network_data_list' in episode.user_data:
                store_network_data(env.evaluation_dir, env.use_real_robot, env.pid, env.episode_counter, reward_total,
                                   episode.user_data['network_data_list'])

        def on_train_result(self, *, trainer, result: dict, **kwargs):
            pass

        def on_postprocess_trajectory(
                self, *, worker: "RolloutWorker", episode: MultiAgentEpisode,
                agent_id: AgentID, policy_id: PolicyID,
                policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
                original_batches: Dict[AgentID, SampleBatch], **kwargs) -> None:
            if args.store_network_data:
                network_data_list = []
                for i in range(len(postprocessed_batch['obs'])):
                    network_data_step = {}
                    for key in postprocessed_batch:
                        network_key = None
                        if key == 'actions':
                            network_key = 'action'
                        elif key == 'obs':
                            network_key = 'observation'
                        elif key in ['action_prob', 'action_logp', 'action_dist_inputs', 'vf_preds', 'fc_1', 'fc_2',
                                     'fc_value_1', 'fc_value_2', 'logits']:
                            network_key = key
                        if network_key is not None:
                            network_data_step[network_key] = postprocessed_batch[key][i]
                    network_data_list.append(network_data_step)
                episode.user_data['network_data_list'] = network_data_list

    tune.register_env(SafeMotionsEnv.__name__,
                      lambda config_args: SafeMotionsEnv(**config_args))
    args.config['callbacks'] = CustomEvaluationCallbacks

    ray.init(dashboard_host="127.0.0.1", include_dashboard=args.use_dashboard, ignore_reinit_error=True,
             num_gpus=args.num_gpus)
    cls = rollout.get_trainable_cls(args.run)
    agent = cls(env=args.env, config=args.config)
    agent.restore(checkpoint_path)

    if checkpoint_config['num_workers'] == 0:
        env = agent.workers.local_worker().env
        if args.store_metrics:
            make_metrics_dir(env.evaluation_dir, args.use_real_robot)
        if args.store_network_data:
            make_network_data_dir(env.evaluation_dir, args.use_real_robot)
        rollout_single_worker_manually()
    else:
        if args.use_real_robot:
            raise ValueError("--use_real_robot requires --num_workers=0")
        rollout_multiple_workers()




