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
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))))

RENDERER = {'opengl': 0,
            'egl': 1,
            'cpu': 2}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_gui', action='store_true', default=False)
    parser.add_argument('--robot_scene', type=int, default=0)
    parser.add_argument('--obstacle_scene', type=int, default=None)
    parser.add_argument('--activate_obstacle_collisions', action='store_true', default=False)
    parser.add_argument('--no_self_collision', action='store_true', default=False)
    parser.add_argument('--online_trajectory_duration', type=float, default=8.0)
    parser.add_argument('--online_trajectory_time_step', type=float, default=0.1)
    parser.add_argument('--use_control_rate_sleep', action='store_true', default=False)
    parser.add_argument('--num_threads_per_worker', type=int, default=1)
    parser.add_argument('--use_thread_for_movement', action='store_true', default=False)
    parser.add_argument('--use_process_for_movement', action='store_true', default=False)
    parser.add_argument('--obstacle_use_computed_actual_values', action='store_true', default=False)
    parser.add_argument('--log_obstacle_data', action='store_true', default=False)
    parser.add_argument('--plot_trajectory', action='store_true', default=False)
    parser.add_argument('--plot_actual_values', action='store_true', default=False)
    parser.add_argument('--plot_acc_limits', action='store_true', default=False)
    parser.add_argument('--switch_gui', action='store_true', default=False)
    parser.add_argument('--pos_limit_factor', type=float, default=1.0)
    parser.add_argument('--vel_limit_factor', type=float, default=1.0)
    parser.add_argument('--acc_limit_factor', type=float, default=1.0)
    parser.add_argument('--jerk_limit_factor', type=float, default=1.0)
    parser.add_argument('--torque_limit_factor', type=float, default=1.0)
    parser.add_argument('--closest_point_safety_distance', type=float, default=0.05)
    parser.add_argument('--target_point_cartesian_range_scene', type=int, default=None)
    parser.add_argument('--target_point_sequence', type=int, default=1)
    parser.add_argument('--target_point_use_actual_position', action='store_true', default=False)
    parser.add_argument('--collision_check_time', type=float, default=None)
    parser.add_argument('--check_braking_trajectory_collisions', action='store_true', default=False)
    parser.add_argument('--check_braking_trajectory_torque_limits', action='store_true', default=False)
    parser.add_argument('--plot_joint', type=json.loads, default=None)
    parser.add_argument("--logging_level", default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--use_real_robot', action='store_true', default=False)
    parser.add_argument('--real_robot_debug_mode', action='store_true', default=False)
    parser.add_argument('--render', action='store_true', default=False,
                        help="If set, videos of the generated episodes are recorded.")
    parser.add_argument("--renderer", default='opengl', choices=['opengl', 'egl', 'cpu'])
    parser.add_argument('--camera_angle', type=int, default=0)
    parser.add_argument('--video_frame_rate', type=float, default=None)
    parser.add_argument('--video_height', type=int, default=None)
    parser.add_argument('--video_dir', type=str, default=None)
    parser.add_argument('--fixed_video_filename', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--solver_iterations', type=int, default=None)

    args = parser.parse_args()

    logging.basicConfig()
    logging.getLogger().setLevel(args.logging_level)

    if args.render and args.renderer == 'egl':
        os.environ['MESA_GL_VERSION_OVERRIDE'] = '3.3'
        os.environ['MESA_GLSL_VERSION_OVERRIDE'] = '330'

    if args.num_threads_per_worker > 0:
        os.environ['OMP_NUM_THREADS'] = str(args.num_threads_per_worker)

    from safemotions.envs.safe_motions_env import SafeMotionsEnv

    seed = args.seed  # None or an integer (for debugging purposes)
    if seed is not None:
        np.random.seed(seed)
    experiment_name = "random_agent"

    use_real_robot = args.use_real_robot
    real_robot_debug_mode = args.real_robot_debug_mode
    # if true, the joint commands are not send to the real robot but all other computations stay the same.
    use_gui = args.use_gui  # if true, the physics simulation is visualized by a GUI (requires an X server)
    switch_gui = args.switch_gui  # True: show background calculations instead of actual robot movements
    control_time_step = None
    # None: use default; if provided, the trajectory timestep must be a multiple of the control_time_step
    use_control_rate_sleep = args.use_control_rate_sleep
    # whether to slow down the simulation such that it is synchronized with the control rate;
    # not required if the simulation is slower anyway
    use_thread_for_movement = args.use_thread_for_movement
    use_process_for_movement = args.use_process_for_movement
    # if true, the control rate is more accurate but actual values have to be computed in advance
    obstacle_use_computed_actual_values = args.obstacle_use_computed_actual_values
    # use computed instead of measured actual values to update the obstacle wrapper -> can be computed in advance
    # -> required to use a separate thread for real-time execution

    render_video = args.render
    # whether to render a video
    camera_angle = args.camera_angle
    # camera_angle for video rendering; predefined settings are defined in video_recording.py

    pos_limit_factor = args.pos_limit_factor
    vel_limit_factor = args.vel_limit_factor
    acc_limit_factor = args.acc_limit_factor
    jerk_limit_factor = args.jerk_limit_factor
    torque_limit_factor = args.torque_limit_factor
    # the maximum values for the position, velocity, acceleration, jerk and torque for each joint are multiplied
    # with the factors defined above

    robot_scene = args.robot_scene
    # 0: one iiwa robot, 1: two iiwa robots, 2: three iiwa robots, 3: armar6, 4: armar6_continuous, 5: armar6_4

    if robot_scene == 0:
        num_joints = 7
    elif robot_scene == 1:
        num_joints = 14
    elif robot_scene == 2:
        num_joints = 21
    elif robot_scene == 3 or robot_scene == 4:
        num_joints = 17
    elif robot_scene == 5:
        num_joints = 33
    else:
        raise ValueError("robot_scene " + str(robot_scene) + " not defined")

    plot_trajectory = args.plot_trajectory
    save_trajectory_plot = False
    if args.plot_joint is None:
        plot_joint = [True] * num_joints
    else:
        plot_joint = args.plot_joint
    # selects the joints to be plotted;
    plot_acc_limits = args.plot_acc_limits
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

    acc_limit_factor_braking = 1.0  # for brake trajectories, relative to acc_limit_factor
    jerk_limit_factor_braking = 1.0  # for brake trajectories, relative to the corresponding maximum jerk

    position_controller_time_constants = None
    # time constants in seconds given per joint; calculate the expected actual values by
    # modelling the behaviour of the trajectory controller without reading actual values from sensor data

    online_trajectory_duration = args.online_trajectory_duration  # duration of each episode in seconds
    online_trajectory_time_step = args.online_trajectory_time_step  # time step between network predictions in seconds

    if args.obstacle_scene is None:
        # 0: no obstacles, 1: table only; 2: Table + walls, 3: table + walls + monitor (no pivot),
        # 4: table + walls + monitor (pivot)
        if robot_scene == 0:
            obstacle_scene = 3
        elif robot_scene < 3:
            obstacle_scene = 1
        else:
            obstacle_scene = 0
    else:
        obstacle_scene = args.obstacle_scene

    activate_obstacle_collisions = args.activate_obstacle_collisions
    # whether the physics engine should allow collisions between the robot and the obstacles
    no_self_collision = args.no_self_collision
    # whether the physics engine should ignore self collisions
    check_braking_trajectory_collisions = args.check_braking_trajectory_collisions
    # collision avoidance by calculating an alternative safe behavior
    collision_check_time = args.collision_check_time
    # time in seconds between collision checks; the trajectory timestep should be a multiple of this time
    check_braking_trajectory_closest_points = True
    # True: Closed points are checked (including self-collision if activated);
    closest_point_safety_distance = args.closest_point_safety_distance
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
    check_braking_trajectory_torque_limits = args.check_braking_trajectory_torque_limits
    # torque constraint adherence by checking the actual torque of an alternative safe behavior

    use_target_points = True  # activate a reaching task with random target points
    if robot_scene < 3:
        target_link_name = "iiwa_link_7"
        # name of the target link for target point reaching
        target_link_offset = [0, 0, 0.126]
        # relative offset between the frame of the target link and the target link point
    else:
        target_link_name = "hand_fixed"
        # name of the target link for target point reaching
        target_link_offset = [0.03, 0, 0.135]
        # relative offset between the frame of the target link and the target link point

    if args.target_point_cartesian_range_scene is None:
        target_point_cartesian_range_scene = robot_scene
        # different settings for the cartesian range of the target point as defined
        # in collision_torque_limit_prevention.py
        # 0: Cartesian range used for a single robot, 1: Cartesian range used for two robots,
        # 2: Cartesian range used for three robots; 3: Cartesian range for armar6
    else:
        target_point_cartesian_range_scene = args.target_point_cartesian_range_scene
    target_point_relative_pos_scene = 0
    # select a setting for the the minimum and maximum relative position between a target point and the target link
    # (required to normalize observations)
    target_point_radius = 0.065
    # a target point is considered as reached if the distance to the target point is smaller than the specified radius
    target_point_sequence = args.target_point_sequence
    # 0: one target point for each robot (T_S), 1: alternating target points (T_A),
    # 2: a single target point for all robots (T_O)
    target_point_use_actual_position = args.target_point_use_actual_position
    # True: Use the actual position values to check whether a target point is reached, False: Use setpoints
    target_point_reached_reward_bonus = 5  # reward bonus if a target point is reached
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

    punish_adaptation = False  # optional adaptation penalty
    adaptation_max_punishment = 0.5  # maximum adaptation penalty if the actual action is replaced by a braking action

    punish_end_min_distance = False
    # whether to add a penalty based on the minimum distance between links or the minimum distance to an obstacle
    # at the end of the decision step
    end_min_distance_max_threshold = 0.05  # distances greater than this threshold are not punished
    end_min_distance_max_punishment = 1

    punish_end_max_torque = False
    # whether to add a penalty based on the maximum (relative) torque at the end of the decision step
    end_max_torque_min_threshold = 0.9  # torques smaller than this threshold are not punished
    end_max_torque_max_punishment = 1

    punish_braking_trajectory_min_distance = False
    # whether to add a penalty based on the minimum distance between the braking trajectory and
    # an observed link or obstacle
    braking_trajectory_min_distance_max_threshold = 0.05

    punish_braking_trajectory_max_torque = False
    # whether to add a penalty based on the maximum torque during the simulation of the braking trajectory
    braking_trajectory_max_torque_min_threshold = 0.8

    braking_trajectory_max_punishment = 1
    # the maximum punishment per step caused by punishing either the minimum distance or the maximum torque

    solver_iterations = args.solver_iterations
    # the maximum number of iterations that the physics solver performs at each time step, None -> 150
    random_agent = True  # whether to use random actions

    env_config = dict(experiment_name=experiment_name,
                      use_gui=use_gui, switch_gui=switch_gui, use_control_rate_sleep=use_control_rate_sleep,
                      use_thread_for_movement=use_thread_for_movement,
                      use_process_for_movement=use_process_for_movement,
                      control_time_step=control_time_step,
                      render_video=render_video, camera_angle=camera_angle,
                      renderer=RENDERER[args.renderer],
                      video_frame_rate=args.video_frame_rate,
                      video_height=args.video_height,
                      video_dir=args.video_dir,
                      fixed_video_filename=args.fixed_video_filename,
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
                      activate_obstacle_collisions=activate_obstacle_collisions,
                      no_self_collision=no_self_collision,
                      observed_link_point_scene=observed_link_point_scene,
                      visualize_bounding_spheres=visualize_bounding_spheres,
                      obstacle_use_computed_actual_values=obstacle_use_computed_actual_values,
                      check_braking_trajectory_collisions=check_braking_trajectory_collisions,
                      collision_check_time=collision_check_time,
                      check_braking_trajectory_observed_points=check_braking_trajectory_observed_points,
                      check_braking_trajectory_closest_points=check_braking_trajectory_closest_points,
                      check_braking_trajectory_torque_limits=check_braking_trajectory_torque_limits,
                      closest_point_safety_distance=closest_point_safety_distance,
                      observed_point_safety_distance=observed_point_safety_distance,
                      use_target_points=use_target_points,
                      target_point_cartesian_range_scene=target_point_cartesian_range_scene,
                      target_point_relative_pos_scene=target_point_relative_pos_scene,
                      target_link_name=target_link_name,
                      target_link_offset=target_link_offset,
                      m_prev=0,
                      obs_add_target_point_pos=obs_add_target_point_pos,
                      obs_add_target_point_relative_pos=obs_add_target_point_relative_pos,
                      punish_action=punish_action,
                      action_punishment_min_threshold=action_punishment_min_threshold,
                      action_max_punishment=action_max_punishment,
                      punish_adaptation=punish_adaptation,
                      adaptation_max_punishment=adaptation_max_punishment,
                      punish_end_min_distance=punish_end_min_distance,
                      end_min_distance_max_threshold=end_min_distance_max_threshold,
                      end_min_distance_max_punishment=end_min_distance_max_punishment,
                      punish_end_max_torque=punish_end_max_torque,
                      end_max_torque_min_threshold=end_max_torque_min_threshold,
                      end_max_torque_max_punishment=end_max_torque_max_punishment,
                      punish_braking_trajectory_min_distance=punish_braking_trajectory_min_distance,
                      braking_trajectory_min_distance_max_threshold=braking_trajectory_min_distance_max_threshold,
                      braking_trajectory_max_punishment=braking_trajectory_max_punishment,
                      punish_braking_trajectory_max_torque=punish_braking_trajectory_max_torque,
                      braking_trajectory_max_torque_min_threshold=braking_trajectory_max_torque_min_threshold,
                      target_point_radius=target_point_radius,
                      target_point_sequence=target_point_sequence,
                      target_point_reached_reward_bonus=target_point_reached_reward_bonus,
                      target_point_use_actual_position=target_point_use_actual_position,
                      target_point_reward_factor=target_point_reward_factor,
                      seed=seed, solver_iterations=solver_iterations, logging_level=args.logging_level,
                      random_agent=random_agent)

    env = SafeMotionsEnv(**env_config)
    num_episodes = args.episodes
    episode_computation_time_list = []
    start = time.time()

    for i in range(num_episodes):
        done = False
        step = 0
        env.reset()
        sum_of_rewards = 0
        start_episode_timer = time.time()

        while not done:
            if render_video:
                import pybullet as p
                keyBoardEvents = p.getKeyboardEvents()
                tKey = ord('t')

                if (tKey in keyBoardEvents) and (keyBoardEvents[tKey] & p.KEY_WAS_TRIGGERED):
                    print('Debug Camera:')
                    debugCamera = p.getDebugVisualizerCamera()
                    print('Yaw: ' + str(debugCamera[8]))
                    print('Pitch: ' + str(debugCamera[9]))
                    print('Distance: ' + str(debugCamera[10]))
                    print('Target: ' + str(debugCamera[11]))
            obs, reward, done, info = env.step(action=None)
            step += 1
            sum_of_rewards += reward

        end_episode_timer = time.time()
        episode_computation_time = end_episode_timer - start_episode_timer
        if env.precomputation_time is not None:
            logging.info("Episode %s took %s seconds. Trajectory duration: %s seconds. Control phase: % seconds.",
                         i + 1, episode_computation_time,
                         step * online_trajectory_time_step,
                         episode_computation_time - env.precomputation_time
                         )
        else:
            logging.info("Episode %s took %s seconds. Trajectory duration: %s seconds.", i + 1,
                         episode_computation_time,
                         step * online_trajectory_time_step)
        episode_computation_time_list.append(episode_computation_time)
        logging.info("Reward: %s", sum_of_rewards)

    end = time.time()
    env.close()
    logging.info("Computed %s episode(s) in %s seconds.", len(episode_computation_time_list), end - start)
    logging.info("Mean computation time: %s seconds, Max computation time: %s seconds.",
                 np.mean(episode_computation_time_list),
                 np.max(episode_computation_time_list))
