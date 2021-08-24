# Learning Collision-free and Torque-limited Robot Trajectories based on Alternative Safe Behaviors 
[![arXiv](https://img.shields.io/badge/arXiv-2103.03793-B31B1B)](https://arxiv.org/abs/2103.03793)
[![PyPI version](https://img.shields.io/pypi/v/safemotions)](https://pypi.python.org/pypi/safemotions)
[![PyPI license](https://img.shields.io/pypi/l/safemotions)](https://pypi.python.org/pypi/safemotions)
[![GitHub issues](https://img.shields.io/github/issues/translearn/safemotions)](https://github.com/translearn/safemotions/issues/)
[![PyPI download month](https://img.shields.io/pypi/dm/safeMotions)](https://pypi.python.org/pypi/safemotions/) <br>
This python package provides the code to learn torque-limited and collision-free robot trajectories without exceeding limits on the position, velocity, acceleration and jerk of each robot joint.

![safemotions_picture](https://user-images.githubusercontent.com/51738372/116555683-f32d7680-a8fc-11eb-8cce-b01931c6ba58.png)

## Installation

The package can be installed by running

    pip install safemotions  # for safe trajectory generation only or
    pip install safemotions[train]  # to include dependencies required to train and evaluate neural networks.

## Trajectory generation &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/translearn/notebooks/blob/main/safemotions_random_agent_demo.ipynb)

To generate a random trajectory with a single robot run

    python -m safemotions.random_agent --use_gui --check_braking_trajectory_collisions --check_braking_trajectory_torque_limits --torque_limit_factor=0.6 --plot_trajectory

For a demonstration scenario with two robots run

    python -m safemotions.random_agent --use_gui --check_braking_trajectory_collisions --robot_scene=1

Collision-free trajectories for three robots can be generated by running

    python -m safemotions.random_agent --use_gui --check_braking_trajectory_collisions --robot_scene=2


## Pretrained networks &nbsp; [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/translearn/notebooks/blob/main/safemotions_trained_networks_demo.ipynb)

Various pretrained networks for reaching randomly sampled target points are provided. \
Make sure you use ray==1.4.1 to open the pretrained networks.  

### Industrial robots 
To generate and plot trajectories for a reaching task with a single industrial robot run

```bash
python -m safemotions.evaluate --checkpoint=industrial/one_robot/collision --use_gui --plot_trajectory --plot_actual_torques
```
Trajectories for two and three industrial robots with alternating target points can be generated by running

```bash
python -m safemotions.evaluate --checkpoint=industrial/two_robots/collision/alternating --use_gui 
```
and
```bash
python -m safemotions.evaluate --checkpoint=industrial/three_robots/collision/alternating --use_gui 
```

### Humanoid robots 

<table width="100%">
    <thead>
        <tr>
            <th style="text-align:center; width: 36%"></th>
            <th style="text-align:center; width: 32%">ARMAR 6</th>
            <th style="text-align:center; width: 32%">ARMAR 6x4</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td style="text-align:center;"></td>
            <td style="text-align:center;"><img src="https://user-images.githubusercontent.com/51738372/130495206-be360e87-2444-4481-86eb-44df5c949880.png" width="300"></td>
           <td style="text-align:center;"><img src="https://user-images.githubusercontent.com/51738372/130494311-0c5e0265-30fc-4a54-962d-a853f16d7cbc.png" width="300"></td>
        </tr>
        <tr>
            <td style="text-align:left;">Alternating target points </td>
            <td style="text-align:left"> --checkpoint=humanoid/armar6/collision/alternating
            </td>
            <td style="text-align:left"> --checkpoint=humanoid/armar6_x4/collision/alternating
            </td>
        </tr>
        <tr>
            <td style="text-align:left;">Simultaneous target points </td>
            <td style="text-align:left"> --checkpoint=humanoid/armar6/collision/simultaneous
            </td>
            <td style="text-align:left"> --checkpoint=humanoid/armar6_x4/collision/simultaneous
            </td>
        </tr>
         <tr>
            <td style="text-align:left;">Single target point </td>
            <td style="text-align:left"> --checkpoint=humanoid/armar6/collision/single
            </td>
            <td style="text-align:left"> --checkpoint=humanoid/armar6_x4/collision/single
            </td>
        </tr>
    </tbody>
</table>



## Training

Networks can also be trained from scratch. For instance, a reaching task with a single robot can be learned by running 
```bash
python -m safemotions.train --logdir=safemotions_training --name=industrial_one_robot_collision --robot_scene=0 --online_trajectory_time_step=0.1 --hidden_layer_activation=swish --online_trajectory_duration=8.0 --obstacle_scene=3 --use_target_points --target_point_sequence=0 --target_point_cartesian_range_scene=0 --target_link_offset="[0, 0, 0.126]" --target_point_radius=0.065 --obs_add_target_point_pos --obs_add_target_point_relative_pos --check_braking_trajectory_collisions --closest_point_safety_distance=0.01 --acc_limit_factor_braking=1.0 --jerk_limit_factor_braking=1.0 --punish_action --action_punishment_min_threshold=0.95 --action_max_punishment=0.4  --target_point_reached_reward_bonus=5  --pos_limit_factor=1.0 --vel_limit_factor=1.0 --acc_limit_factor=1.0 --jerk_limit_factor=1.0 --torque_limit_factor=1.0 --punish_braking_trajectory_min_distance --braking_trajectory_min_distance_max_threshold=0.05 --braking_trajectory_max_punishment=0.5 --last_layer_activation=tanh --solver_iterations=50 --normalize_reward_to_initial_target_point_distance --collision_check_time=0.033 --iterations_per_checkpoint=50 --time=200
```

## Publication
The corresponding publication is available at [https://arxiv.org/abs/2103.03793](https://arxiv.org/abs/2103.03793).

[![Video](https://yt-embed.herokuapp.com/embed?v=5YpUhMx1xZM
)](https://www.youtube.com/watch?v=5YpUhMx1xZM)


## Disclaimer

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.