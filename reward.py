import numpy as np
import torch

def abs_reward(loc_diff, center=5.0):
    deviation = abs(loc_diff - center)
    return -deviation if loc_diff > center else deviation

def calculate_reward(observation, action, done, progress_diff, velocity, 
                      aim_point_diff, location_diff=None, finished=False):
    """
    Calculate the reward based on the observation and action taken.
    @observation: image of the race
    @action: dict containing the action taken
    @done: boolean indicating if the episode is done
    @progress_diff: difference in progress from the last step
    @velocity: current velocity of the kart
    @agent_initiated_rescue_flag: boolean indicating if the agent initiated a rescue (kart is stuck or crushed)
    @aim_point_diff: difference in aim point from the track center in the forward direction (15m in front by default)
    @location_diff: difference in location from the track center
    return: calculated reward
    """
    reward = 0
    
    reward += progress_diff * 60
    
    target_velocity = 27.0
    if velocity < target_velocity * 0.2:
        reward -= 0.5 * (target_velocity - velocity) / target_velocity
    
    if not done:
        reward -= 0.01
    
    if action.rescue == True:
        reward -= 8
    
    # if action is not None:
    #     if action.brake and velocity < target_velocity * 0.5:
    #         reward -= 0.5 
    reward += abs_reward(aim_point_diff, 0.5) * 2.0
    
    if location_diff is not None:
        location_diff = np.clip(location_diff, 0, 10.0)
        reward += abs_reward(location_diff, 5)

    # Terminal states
    if done:
        reward += 20.0
    elif finished:
        reward -= 20.0
    return np.clip(reward, -20.0, 20.0)

