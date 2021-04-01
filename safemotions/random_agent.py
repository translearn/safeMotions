#!/usr/bin/env python

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import numpy as np
import argparse
import time
import logging
import os
import sys
import inspect
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))
from safemotions.envs.safe_motions_env import SafeMotionsEnv


if __name__ == '__main__':
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--robot_scene', type=int, default=0)
    parser.add_argument('--obstacle_scene', type=int, default=None)
    parser.add_argument('--use_control_rate_sleep', action='store_true', default=False)
    parser.add_argument('--use_thread_for_control_rate_sleep', action='store_true', default=False)
    parser.add_argument('--obstacle_use_computed_actual_values', action='store_true', default=False)
    parser.add_argument('--log_obstacle_data', action='store_true', default=False)
    parser.add_argument('--plot_trajectory', action='store_true', default=False)
    parser.add_argument('--plot_actual_values', action='store_true', default=False)
    args = parser.parse_args()

    seed = None  # None or an integer (for debugging purposes)
    if seed is not None:
        np.random.seed(seed)
    experiment_name = "random_agent"

    use_real_robot = False
    real_robot_debug_mode = False
    # if true, the joint commands are not send to the real robot but all other computations stay the same.
    use_gui = True
    control_time_step = None
    # None: use default; if provided, the trajectory timestep must be a multiple of the control_time_step
    use_control_rate_sleep = args.use_control_rate_sleep
    # whether to slow down the simulation such that it is synchronized with the control rate;
    # not required if the simulation is slower anyway
    use_thread_for_control_rate_sleep = args.use_thread_for_control_rate_sleep
    # if true, the control rate is more accurate but actual values have to be computed in advance
    obstacle_use_computed_actual_values = args.obstacle_use_computed_actual_values
    # use computed instead of measured actual values to update the obstacle wrapper -> can be computed in advance
    # -> required to use a separate thread for real-time execution

    render_video = False
    # whether to render a video
    camera_angle = 1
    # camera_angle for video rendering; predefined settings are defined in video_recording.py

    pos_limit_factor = 1.0
    vel_limit_factor = 1.0
    acc_limit_factor = 1.0
    jerk_limit_factor = 1.0
    torque_limit_factor = 1.0
    # the maximum values for the position, velocity, acceleration, jerk and torque for each joint are multiplied
    # with the factors defined above

    robot_scene = args.robot_scene
    # 0: one iiwa robot, 1: two iiwa robots, 2: three iiwa robots

    if robot_scene == 0:
        num_robots = 1
    if robot_scene == 1:
        num_robots = 2
    if robot_scene == 2:
        num_robots = 3

    plot_trajectory = args.plot_trajectory
    save_trajectory_plot = False
    plot_joint = [True, True, True, True, True, True, True] * num_robots
    # selects the joints to be plotted;
    plot_acc_limits = False
    plot_actual_values = args.plot_actual_values
    plot_computed_actual_values = False
    plot_actual_torques = True
    plot_time_limits = None
    log_obstacle_data = args.log_obstacle_data
    # plots of obstacle date are only available if log_obstacle_data is True
    save_obstacle_data = False
    # store the data of the obstacle wrapper as a pickle object; saves the trajectory plot as well
    store_actions = False
    # save the list of predicted actions

    acc_limit_factor_braking = 0.75  # for brake trajectories, relative to acc_limit_factor
    jerk_limit_factor_braking = 0.75  # for brake trajectories, relative to the corresponding maximum jerk

    position_controller_time_constants = [0.030, 0.030, 0.030, 0.030, 0.030, 0.030, 0.030] * num_robots
    # time constants in seconds given per joint; calculate the expected actual values by
    # modelling the behaviour of the trajectory controller without reading actual values from sensor data

    online_trajectory_duration = 8  # time in seconds
    online_trajectory_time_step = 0.1  # in seconds

    if args.obstacle_scene is None:
        # 0: no obstacles, 1: table only; 2: Table + walls, 3: table + walls + monitor (no pivot),
        # 4: table + walls + monitor (pivot)
        if robot_scene == 0:
            obstacle_scene = 3
        else:
            obstacle_scene = 1
    else:
        obstacle_scene = args.obstacle_scene

    use_braking_trajectory_method = True
    # collision avoidance by calculating an alternative safe behavior
    collision_check_time = 0.05
    # time in seconds between collision checks; the trajectory timestep should be a multiple of this time
    check_braking_trajectory_closest_points = True
    # True: Closed points are checked (including self-collision if activated);
    closest_point_safety_distance = 0.05
    # minimum distance that should be guaranteed by the collision avoidance method for closest points
    check_braking_trajectory_observed_points = False
    # True: Observed points are checked;
    observed_point_safety_distance = 0.05
    # minimum distance that should be guaranteed by the collision avoidance method for observed points
    observed_link_point_scene = 0
    # 0: no observed link points, 1: observe link 6 and 7 with a single sphere,
    # 2: observe the robot body starting from link 3 with in total 6 spheres
    visualize_bounding_spheres = False
    # whether to visualize the bounding spheres of observed link points, requires log_obstacle_data to be True
    check_braking_trajectory_torque_limits = True
    # torque constraint adherence by checking the actual torque of an alternative safe behavior

    use_target_points = True  # activate a reaching task with random target points
    target_link_name = "iiwa_link_7"  # name of the target link for target point reaching
    target_link_offset = [0, 0, 0.126]  # relative offset between the frame of the target link and the target link point
    target_point_cartesian_range_scene = num_robots - 1
    # different settings for the cartesian range of the target point as defined in obstacle_torque_prevention.py
    # 0: Cartesian range used for a single robot, 1: Cartesian range used for two robots,
    # 2: Cartesian range used for three robots
    target_point_radius = 0.065
    # a target point is considered as reached if the distance to the target point is smaller than the specified radius
    target_point_sequence = 1  # 0: target points for all robots (T_S), 1: alternating target points (T_A)
    target_point_reached_reward_bonus = 5  # reward bonus if a target point is reached
    target_point_max_jerk_punishment = 1.0  # the jerk punishment is multiplied by this factor
    target_point_reward_factor = 1.0  # the target point reward is multiplied by this factor

    if use_target_points:
        obs_add_target_point_pos = True  # add the Cartesian position of the target point to the observation
        obs_add_target_point_relative_pos = True
        # add the difference between the target point and the target link point in Cartesian space (for each coordinate)
    else:
        obs_add_target_point_pos = False
        obs_add_target_point_relative_pos = False

    punish_action = True  # optional action penalty
    action_punishment_min_threshold = 0.9  # (joint) actions greater than this threshold are punished
    action_max_punishment = 1  # maximum punishment if the absolute value of the highest (joint) action is 1

    punish_braking_trajectory_min_distance = False
    # whether to add a penalty based on the minimum distance between the braking trajectory and
    # an observed link or obstacle
    braking_trajectory_min_distance_max_threshold = 0.05
    braking_trajectory_min_distance_max_punishment = 1

    punish_braking_trajectory_max_torque = False
    # whether to add a penalty based on the maximum torque during the simulation of the braking trajectory
    braking_trajectory_max_torque_min_threshold = 0.8
    braking_trajectory_max_torque_max_punishment = 0.5

    random_agent = True  # whether to use random actions

    env_config = dict(env="ObstacleTorqueEnv", experiment_name=experiment_name,
                      use_gui=use_gui, use_control_rate_sleep=use_control_rate_sleep,
                      use_thread_for_control_rate_sleep=use_thread_for_control_rate_sleep,
                      control_time_step=control_time_step,
                      render_video=render_video, camera_angle=camera_angle,
                      use_real_robot=use_real_robot,
                      real_robot_debug_mode=real_robot_debug_mode,
                      pos_limit_factor=pos_limit_factor,
                      vel_limit_factor=vel_limit_factor,
                      acc_limit_factor=acc_limit_factor,
                      jerk_limit_factor=jerk_limit_factor,
                      torque_limit_factor=torque_limit_factor,
                      plot_trajectory=plot_trajectory, save_trajectory_plot=save_trajectory_plot,
                      plot_joint=plot_joint, plot_acc_limits=plot_acc_limits,
                      plot_actual_values=plot_actual_values,
                      plot_computed_actual_values=plot_computed_actual_values,
                      plot_actual_torques=plot_actual_torques,
                      plot_time_limits=plot_time_limits,
                      log_obstacle_data=log_obstacle_data,
                      save_obstacle_data=save_obstacle_data,
                      store_actions=store_actions,
                      acc_limit_factor_braking=acc_limit_factor_braking,
                      jerk_limit_factor_braking=jerk_limit_factor_braking,
                      online_trajectory_duration=online_trajectory_duration,
                      online_trajectory_time_step=online_trajectory_time_step,
                      position_controller_time_constants=position_controller_time_constants,
                      robot_scene=robot_scene,
                      obstacle_scene=obstacle_scene,
                      observed_link_point_scene=observed_link_point_scene,
                      visualize_bounding_spheres=visualize_bounding_spheres,
                      obstacle_use_computed_actual_values=obstacle_use_computed_actual_values,
                      use_braking_trajectory_method=use_braking_trajectory_method,
                      collision_check_time=collision_check_time,
                      check_braking_trajectory_observed_points=check_braking_trajectory_observed_points,
                      check_braking_trajectory_closest_points=check_braking_trajectory_closest_points,
                      check_braking_trajectory_torque_limits=check_braking_trajectory_torque_limits,
                      closest_point_safety_distance=closest_point_safety_distance,
                      observed_point_safety_distance=observed_point_safety_distance,
                      use_target_points=use_target_points,
                      target_point_cartesian_range_scene=target_point_cartesian_range_scene,
                      obs_add_target_point_pos=obs_add_target_point_pos,
                      obs_add_target_point_relative_pos=obs_add_target_point_relative_pos,
                      punish_action=punish_action,
                      action_punishment_min_threshold=action_punishment_min_threshold,
                      action_max_punishment=action_max_punishment,
                      target_link_name=target_link_name,
                      target_link_offset=target_link_offset,
                      punish_braking_trajectory_min_distance=punish_braking_trajectory_min_distance,
                      braking_trajectory_min_distance_max_threshold=braking_trajectory_min_distance_max_threshold,
                      braking_trajectory_min_distance_max_punishment=braking_trajectory_min_distance_max_punishment,
                      punish_braking_trajectory_max_torque=punish_braking_trajectory_max_torque,
                      braking_trajectory_max_torque_min_threshold=braking_trajectory_max_torque_min_threshold,
                      braking_trajectory_max_torque_max_punishment=braking_trajectory_max_torque_max_punishment,
                      target_point_radius=target_point_radius,
                      target_point_sequence=target_point_sequence,
                      target_point_reached_reward_bonus=target_point_reached_reward_bonus,
                      target_point_max_jerk_punishment=target_point_max_jerk_punishment,
                      target_point_reward_factor=target_point_reward_factor,
                      seed=seed, random_agent=random_agent)

    env = SafeMotionsEnv(**env_config)
    num_episodes = 20

    for _ in range(num_episodes):
        done = False
        step = 0
        env.reset()
        sum_of_rewards = 0
        start = time.time()

        while not done:
            obs, reward, done, info = env.step(action=None)
            step += 1
            sum_of_rewards += reward

        end = time.time()
        logging.info("Last episode took %s seconds. Trajectory duration: %s seconds.", end - start,
                     step * online_trajectory_time_step)
        logging.info("Reward: %s", sum_of_rewards)

    env.close()
