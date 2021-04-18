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
import datetime
import time
import logging
import numpy as np
import errno
from ray.rllib import rollout
from ray import tune
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.append(os.path.dirname(current_dir))
from safemotions.envs.safe_motions_env import SafeMotionsEnv

# Termination reason
TERMINATION_UNSET = -1
TERMINATION_SUCCESS = 0
TERMINATION_JOINT_LIMITS = 1
TERMINATION_TRAJECTORY_LENGTH = 2


def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


def rollout_manually(agent, evaluation_dir):
    env = agent.workers.local_worker().env
    episodes_sampled = 0
    timestamp = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
    experiment_name = os.path.join(env_config['experiment_name'], timestamp)

    if args.store_metrics:
        evaluation_dir = os.path.join(evaluation_dir, "trajectory_logs")
        if args.use_real_robot:
            evaluation_dir = os.path.join(evaluation_dir, "real")
        else:
            evaluation_dir = os.path.join(evaluation_dir, "sim")
        if not os.path.exists(os.path.join(evaluation_dir, experiment_name)):
            try:
                os.makedirs(os.path.join(evaluation_dir, experiment_name))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        with open(os.path.join(evaluation_dir, experiment_name, 'config.json'), 'w') as f:
            f.write(json.dumps(vars(args)))
            f.flush()

    episode_computation_time_list = []
    start = time.time()

    while True:
        if args.episodes:
            if episodes_sampled >= args.episodes:
                break

        if args.use_real_robot:
            metric_file = str(episodes_sampled) + '_' + timestamp + "_real" + ".json"
        else:
            metric_file = str(episodes_sampled) + '_' + timestamp + "_sim" + ".json"

        obs = env.reset()
        done = False
        reward_total = 0.0
        episode_info = {}
        steps = -1
        start_episode_timer = time.time()
        while not done:
            steps = steps + 1
            action = agent.compute_action(obs)

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
            logging.debug("Observation %s: %s", steps, obs)
            logging.debug("Action %s: %s", steps, action)
            logging.debug("Reward %s: %s", steps, reward)
            obs = next_obs

        end_episode_timer = time.time()
        episode_computation_time = end_episode_timer - start_episode_timer
        logging.info("Last episode took %s seconds", episode_computation_time)
        episode_computation_time_list.append(episode_computation_time)
        logging.info("Trajectory duration: %s seconds", (steps + 1) * env.trajectory_time_step)

        if args.store_metrics:
            episode_info['reward'] = float(reward_total)
            episode_info['episode_length'] = int(info['episode_length'])
            episode_info['trajectory_length'] = int(info['trajectory_length'])
            for key, value in info.items():
                if key.startswith("obstacles"):
                    episode_info[key] = value

            for key, value in episode_info['max'].items():
                episode_info['max'][key] = float(np.max(np.array(value)))
            for key, value in episode_info['average'].items():
                episode_info['average'][key] = float(np.mean(np.array(value)))
            for key, value in episode_info['min'].items():
                episode_info['min'][key] = float(np.min(np.array(value)))

            if info['termination_reason'] == TERMINATION_SUCCESS:
                episode_info['success_rate'] = int(1.0)
            else:
                episode_info['success_rate'] = int(0.0)

            if info['termination_reason'] == TERMINATION_JOINT_LIMITS:
                episode_info['joint_limit_violation_termination_rate'] = int(1.0)
            else:
                episode_info['joint_limit_violation_termination_rate'] = int(0.0)

            with open(os.path.join(evaluation_dir, experiment_name, metric_file), 'w') as f:
                f.write(json.dumps(episode_info, default=np_encoder))
                f.flush()

        episodes_sampled += 1
        logging.info("Episode reward: %s", reward_total)

    env.close()
    end = time.time()
    logging.info("Computed %s episodes in %s seconds.", len(episode_computation_time_list), end - start)
    logging.info("Mean computation time: %s seconds, Max computation time: %s seconds.",
                 np.mean(episode_computation_time_list),
                 np.max(episode_computation_time_list))


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None,
                        help="The name of the evaluation.")
    parser.add_argument('--evaluation_dir', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, required=True,
                        help="Path to the checkpoint for evaluation.")
    parser.add_argument('--episodes', type=int, default=20,
                        help="The number of episodes for evaluation.")
    parser.add_argument('--render', action='store_true', default=False,
                        help="Whether or not to render videos of the rollouts.")
    parser.add_argument('--camera_angle', type=int, default=0)
    parser.add_argument('--use_real_robot', action='store_true', default=None)
    parser.add_argument('--real_robot_debug_mode', dest='real_robot_debug_mode', action='store_true', default=False)
    parser.add_argument('--use_gui', action='store_true', default=False)
    parser.add_argument('--store_metrics', action='store_true', default=False)
    parser.add_argument('--plot_trajectory', action='store_true', default=False)
    parser.add_argument('--save_trajectory_plot', action='store_true', default=False)
    parser.add_argument('--plot_acc_limits', action='store_true', default=False)
    parser.add_argument('--plot_actual_values', action='store_true', default=False)
    parser.add_argument('--plot_actual_torques', action='store_true', default=False)
    parser.add_argument('--plot_computed_actual_values', action='store_true', default=False)
    parser.add_argument('--plot_joint', type=json.loads, default=None)
    parser.add_argument('--store_actions', action='store_true', default=False)
    parser.add_argument('--log_obstacle_data', action='store_true', default=False)
    parser.add_argument('--obstacle_scene', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--random_agent', action='store_true', default=False)
    parser.add_argument('--no_self_collision', action='store_true', default=False)
    parser.add_argument('--use_thread_for_control_rate_sleep', action='store_true', default=False)
    parser.add_argument('--control_time_step', type=float, default=None)
    parser.add_argument('--time_step_fraction_sleep_observation', type=float, default=None)

    args = parser.parse_args()

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
        env_config = checkpoint_config['env_config']

    if args.name is not None:
        env_config['experiment_name'] = args.name

    if args.render:
        env_config.update(render_video=True)
        env_config['camera_angle'] = args.camera_angle
    else:
        env_config.update(render_video=False)

    if args.use_gui:
        env_config.update(use_gui=True)
    else:
        env_config.update(use_gui=False)

    env_config.update(use_real_robot=args.use_real_robot)

    if args.store_actions:
        env_config['store_actions'] = True

    if args.log_obstacle_data:
        env_config['log_obstacle_data'] = True

    if args.save_trajectory_plot:
        env_config['save_trajectory_plot'] = True

    if args.plot_actual_torques:
        env_config['plot_actual_torques'] = True

    if args.use_thread_for_control_rate_sleep:
        env_config['use_thread_for_control_rate_sleep'] = args.use_thread_for_control_rate_sleep

    if args.obstacle_scene is not None:
        env_config['obstacle_scene'] = args.obstacle_scene

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

    if args.real_robot_debug_mode:
        env_config['real_robot_debug_mode'] = True

    if args.no_self_collision:
        env_config['no_self_collision'] = True

    if args.time_step_fraction_sleep_observation is not None:
        env_config['time_step_fraction_sleep_observation'] = args.time_step_fraction_sleep_observation

    if args.control_time_step is not None:
        env_config['control_time_step'] = args.control_time_step

    checkpoint_config['num_workers'] = 0
    checkpoint_config['env_config'] = env_config

    if args.seed is not None:
        checkpoint_config['seed'] = args.seed
        env_config['seed'] = args.seed

    args.config = checkpoint_config
    args.run = "PPO"
    tune.register_env(SafeMotionsEnv.__name__,
                      lambda config_args: SafeMotionsEnv(**config_args))
    args.env = checkpoint_config['env']
    args.out = None
    ray.init()
    cls = rollout.get_trainable_cls(args.run)
    agent = cls(env=args.env, config=args.config)
    agent.restore(checkpoint_path)
    if args.seed is not None:
        np.random.seed(args.seed)
    rollout_manually(agent, evaluation_dir)

