import torch
import numpy as np
import pystk
import torchvision.transforms.functional as TF
from controller import control

class PPODataCollector:
    """Adapter to collect data from PyTux for PPO training"""
    
    def __init__(self, policy, calculate_reward_fn, control_fn, device='cpu'):
        self.policy = policy
        self.calculate_reward_fn = calculate_reward_fn
        self.control_fn = control_fn
        self.device = device
        self.pytux = None
        self.current_track = None
        self.episode_data = None
        
    def initialize(self):
        """Initialize pytux if not already initialized"""
        from utils import PyTux
        if self.pytux is None:
            self.pytux = PyTux()
    
    def _reset_episode_data(self):
        """Reset episode data storage"""
        self.episode_data = {
            'obs': [],
            'actions': [],
            'log_probs': [],
            'rewards': [],
            'dones': [],
            'values': [],
            'episode_reward': 0,
            'last_progress': 0
        }
    
    def collect_episode(self, track, max_frames=1000, verbose=False):
        """Collect episode data using PyTux.rollout"""
        self.initialize()
        self._reset_episode_data()
        self.current_track = track
        # state_recorder = {'steps': 0, 'last_progress': 0, 'total_reward': 0, 'dones': []}

        ppo_data = self.pytux.ppo_rollout(
            track=track, 
            policy=self.policy,
            reward_fn=self.calculate_reward_fn,
            control_fn=self.control_fn,
            max_frames=max_frames,
            verbose=verbose,
        )

        self.episode_data = ppo_data
        
        # print(f"ppo_data['obs']: {ppo_data['obs'].shape}")
        
        # Process the collected data
        if len(self.episode_data['obs']) == 0:
            # No steps collected
            print(f"No steps collected for track {track}, skipping update")
            return {
                'obs': np.array([]),
                'actions': np.array([]),
                'log_probs': np.array([]),
                'rewards': np.array([]),
                'dones': np.array([]),
                'values': np.array([]),
                'last_value': 0,
                'episode_reward': 0,
                'episode_length': 0,
                'final_progress': 0
            }
        # elif len(self.episode_data['obs']) == 1:
        #     # Only one step collected
        #     print(f"Only one step collected for track {track}, skipping update")
        #     return {
        #         'obs': np.array([]),
        #         'actions': np.array([]),
        #         'log_probs': np.array([]),
        #         'rewards': np.array([]),
        #         'dones': np.array([]),
        #         'values': np.array([]),
        #         'last_value': 0,
        #         'episode_reward': 0,
        #         'episode_length': 0,
        #         'final_progress': 0
        #     }
        # Convert to numpy arrays
        
        # Add dummy reward for the last step if rewards length doesn't match
        # while len(self.episode_data['rewards']) < len(self.episode_data['obs']):
        #     self.episode_data['rewards'].append(0.0)
            
        # Add dummy done values if dones length doesn't match
        # while len(state_recorder['dones']) < len(self.episode_data['obs']):
        #     state_recorder['dones'].append(True)
        
        # # Add final state value for GAE calculation
        # with torch.no_grad():
        #     if len(self.episode_data['obs']) > 0:
        #         final_obs = self.episode_data['obs'][-1]
        #         final_obs_tensor = torch.FloatTensor(final_obs).to(self.device)
        #         _, last_value = self.policy(final_obs_tensor.unsqueeze(0))
        #         last_value = last_value.cpu().numpy()[0]
        #     else:
        #         last_value = 0
        
        # Return results
        return self.episode_data
    
    def close(self):
        if self.pytux is not None:
            self.pytux.close()
            self.pytux = None