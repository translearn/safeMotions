#!/usr/bin/env python

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import argparse
import json
import os
import ray
from ray import tune
import multiprocessing
from collections import defaultdict
import sys
import logging
import inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))
from safemotions.envs.safe_motions_env import SafeMotionsEnv

METRIC_OPS = ['sum', 'average', 'max', 'min']

# Termination reason
TERMINATION_UNSET = -1
TERMINATION_SUCCESS = 0
TERMINATION_JOINT_LIMITS = 1
TERMINATION_TRAJECTORY_LENGTH = 2


def on_episode_start(info):
    episode = info['episode']
    for op in METRIC_OPS:
        episode.user_data[op] = defaultdict(list)


def on_episode_step(info):
    episode = info['episode']
    episode_info = episode.last_info_for()
    if episode_info:
        for op in list(episode_info.keys() & METRIC_OPS):
            for k, v in episode_info[op].items():
                episode.user_data[op][k].append(v)


def on_episode_end(info):
    def __apply_op_on_list(operator, data_list):
        if operator == 'sum':
            return sum(data_list)
        elif operator == 'average':
            return sum(data_list) / len(data_list)
        elif operator == 'max':
            return max(data_list)
        elif operator == 'min':
            return min(data_list)

    episode = info['episode']
    episode_info = episode.last_info_for()
    episode_length = episode_info['episode_length']
    trajectory_length = episode_info['trajectory_length']

    for op in METRIC_OPS:
        for k, v in episode.user_data[op].items():
            episode.custom_metrics[k + "_" + op] = __apply_op_on_list(op, episode.user_data[op][k])

    for k, v in episode_info.items():
        if k.startswith("obstacles"):
            episode.custom_metrics[k] = v

    episode.custom_metrics['episode_length'] = float(episode_length)
    episode.custom_metrics['trajectory_length'] = trajectory_length

    if episode_info['termination_reason'] == TERMINATION_SUCCESS:
        episode.custom_metrics['success_rate'] = 1.0
    else:
        episode.custom_metrics['success_rate'] = 0.0

    if episode_info['termination_reason'] == TERMINATION_JOINT_LIMITS:
        episode.custom_metrics['joint_limit_violation_termination_rate'] = 1.0
    else:
        episode.custom_metrics['joint_limit_violation_termination_rate'] = 0.0


def on_train_result(info):
    info['result']['callback_ok'] = True


callbacks = {'on_episode_start': on_episode_start,
             'on_episode_step': on_episode_step,
             'on_episode_end': on_episode_end,
             'on_train_result': on_train_result}


def _make_env_config():

    env_config = {
        'experiment_name': args.name,
        'm_prev': args.m_prev,
        'pos_limit_factor': args.pos_limit_factor,
        'vel_limit_factor': args.vel_limit_factor,
        'acc_limit_factor': args.acc_limit_factor,
        'jerk_limit_factor': args.jerk_limit_factor,
        'torque_limit_factor': args.torque_limit_factor,
        'normalize_reward_to_frequency': args.normalize_reward_to_frequency,
        'online_trajectory_duration': args.online_trajectory_duration,
        'online_trajectory_time_step': args.online_trajectory_time_step,
        'obs_add_target_point_pos': args.obs_add_target_point_pos,
        'obs_add_target_point_relative_pos': args.obs_add_target_point_relative_pos,
        'punish_action': args.punish_action,
        'action_punishment_min_threshold': args.action_punishment_min_threshold,
        'action_max_punishment': args.action_max_punishment,
        'obstacle_scene': args.obstacle_scene,
        'log_obstacle_data': False,
        'use_braking_trajectory_method': args.use_braking_trajectory_method,
        'check_braking_trajectory_torque_limits': args.check_braking_trajectory_torque_limits,
        'closest_point_safety_distance': args.closest_point_safety_distance,
        'use_target_points': args.use_target_points,
        'acc_limit_factor_braking': args.acc_limit_factor_braking,
        'jerk_limit_factor_braking': args.jerk_limit_factor_braking,
        'punish_braking_trajectory_min_distance': args.punish_braking_trajectory_min_distance,
        'braking_trajectory_min_distance_max_threshold': args.braking_trajectory_min_distance_max_threshold,
        'braking_trajectory_min_distance_max_punishment': args.braking_trajectory_min_distance_max_punishment,
        'punish_braking_trajectory_max_torque': args.punish_braking_trajectory_max_torque,
        'braking_trajectory_max_torque_min_threshold': args.braking_trajectory_max_torque_min_threshold,
        'braking_trajectory_max_torque_max_punishment': args.braking_trajectory_max_torque_max_punishment,
        'robot_scene': args.robot_scene,
        'target_point_cartesian_range_scene': args.target_point_cartesian_range_scene,
        'target_point_radius': args.target_point_radius,
        'target_point_sequence': args.target_point_sequence,
        'target_point_reached_reward_bonus': args.target_point_reached_reward_bonus,
        'target_point_reward_factor': args.target_point_reward_factor,
        'target_link_offset': args.target_link_offset,
        'obstacle_use_computed_actual_values': args.obstacle_use_computed_actual_values
    }

    return env_config


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    config = {
        'model': {
            'conv_filters': None,
            'fcnet_hiddens': [256, 128],
            'fcnet_activation': "selu",
            'use_lstm': False,
        },
        'gamma': 0.99,
        'use_gae': True,
        'lambda': 1.0,
        'kl_coeff': 0.2,
        'sample_batch_size': 4096,
        'train_batch_size': 16384,
        'sgd_minibatch_size': 1024,
        'num_sgd_iter': 16,
        'lr': 5e-5,
        'lr_schedule': None,
        'vf_loss_coeff': 1.0,
        'entropy_coeff': 0.00,
        'clip_param': 0.3,
        'vf_clip_param': 10.0,
        'kl_target': 0.01,
        'batch_mode': 'complete_episodes',
    }

    algorithm = "PPO"
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default="default_name")
    parser.add_argument('--logdir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path to a checkpoint if training should be continued.")
    parser.add_argument('--time', type=int, required=True,
                        help="Total time of the training in hours.")
    parser.add_argument('--iterations_per_checkpoint', type=int, default=500,
                        help="The number of training iterations per checkpoint")
    parser.add_argument('--num_workers', type=int, default=None)
    parser.add_argument('--num_gpus', type=int, default=None)
    parser.add_argument('--m-prev', type=int, default=1)
    parser.add_argument('--pos_limit_factor', type=float, default=1.0)
    parser.add_argument('--vel_limit_factor', type=float, default=1.0)
    parser.add_argument('--acc_limit_factor', type=float, default=1.0)
    parser.add_argument('--jerk_limit_factor', type=float, default=1.0)
    parser.add_argument('--torque_limit_factor', type=float, default=1.0)
    parser.add_argument('--normalize_reward_to_frequency', dest='normalize_reward_to_frequency', action='store_true',
                        default=False)
    parser.add_argument('--batch_size_factor', type=float, default=1.0)
    parser.add_argument('--online_trajectory_duration', type=float, default=8.0)
    parser.add_argument('--online_trajectory_time_step', type=float, default=0.1)
    parser.add_argument('--obs_add_target_point_pos', action='store_true', default=False)
    parser.add_argument('--obs_add_target_point_relative_pos', action='store_true', default=False)
    parser.add_argument('--punish_action', dest='punish_action', action='store_true', default=False)
    parser.add_argument('--action_punishment_min_threshold', type=float, default=0.9)
    parser.add_argument('--action_max_punishment', type=float, default=0.5)
    parser.add_argument('--punish_braking_trajectory_min_distance', action='store_true', default=False)
    parser.add_argument('--braking_trajectory_min_distance_max_threshold', type=float, default=0.05)
    parser.add_argument('--braking_trajectory_min_distance_max_punishment', type=float, default=1.0)
    parser.add_argument('--punish_braking_trajectory_max_torque', action='store_true', default=False)
    parser.add_argument('--braking_trajectory_max_torque_min_threshold', type=float, default=0.8)
    parser.add_argument('--braking_trajectory_max_torque_max_punishment', type=float, default=0.5)
    parser.add_argument('--obstacle_scene', type=int, default=0)
    parser.add_argument('--use_braking_trajectory_method', action='store_true', default=False)
    parser.add_argument('--check_braking_trajectory_torque_limits', action='store_true', default=False)
    parser.add_argument('--closest_point_safety_distance', type=float, default=0.1)
    parser.add_argument('--use_target_points', action='store_true', default=False)
    parser.add_argument('--acc_limit_factor_braking', type=float, default=0.75)
    parser.add_argument('--jerk_limit_factor_braking', type=float, default=0.75)
    parser.add_argument('--robot_scene', type=int, default=0)
    parser.add_argument('--target_point_cartesian_range_scene', type=int, default=0)
    parser.add_argument('--target_point_radius', type=float, default=0.065)
    parser.add_argument('--target_point_sequence', type=int, default=0)
    parser.add_argument('--target_point_reached_reward_bonus', type=float, default=0.00)
    parser.add_argument('--target_link_offset', type=json.loads, default="[0, 0, 0.126]")
    parser.add_argument('--target_point_reward_factor', type=float, default=1.0)
    parser.add_argument('--obstacle_use_computed_actual_values', action='store_true', default=False)
    args = parser.parse_args()
    env_name = "SafeMotionsEnv"
    tune.register_env(env_name, lambda config_args: SafeMotionsEnv(**config_args))
    config.update(env=env_name)

    if args.checkpoint is not None:
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
        config['env_config'] = checkpoint_config['env_config']
        config['sample_batch_size'] = checkpoint_config['sample_batch_size']
        config['train_batch_size'] = checkpoint_config['train_batch_size']
        config['sgd_minibatch_size'] = checkpoint_config['sgd_minibatch_size']

    else:
        checkpoint_path = None
        config['env_config'] = _make_env_config()
        config['sample_batch_size'] = int(config['sample_batch_size'] * args.batch_size_factor)
        config['train_batch_size'] = int(config['train_batch_size'] * args.batch_size_factor)
        config['sgd_minibatch_size'] = int(config['sgd_minibatch_size'] * args.batch_size_factor)

    if args.logdir is None:
        experiment_path = config['env_config']['experiment_name']
    else:
        experiment_path = os.path.join(args.logdir, config['env_config']['experiment_name'])

    ray.init()
    config['callbacks'] = callbacks

    if args.num_workers is None:
        config['num_workers'] = int(multiprocessing.cpu_count() * 0.75)
    else:
        config['num_workers'] = args.num_workers

    if args.num_gpus is not None:
        config['num_gpus'] = args.num_gpus

    stop = {'time_total_s': args.time * 3600}

    experiment = {
        experiment_path: {
            'run': algorithm,
            'env': env_name,
            'stop': stop,
            'config': config,
            'checkpoint_freq': args.iterations_per_checkpoint,
            'checkpoint_at_end': True,
            'keep_checkpoints_num': 10,
            'max_failures': 0,
            'restore': checkpoint_path
        }
    }

    tune.run_experiments(experiment)
