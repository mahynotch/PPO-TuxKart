import torch
import numpy as np
import argparse
from ppo_planner import load_model

def test_planner(args):
    from controller import control
    from utils import PyTux
    
    # Load model
    planner = load_model(args.file)
    pytux = PyTux()
    
    results = {}
    
    # Define a wrapper to use our PPO planner with PyTux's rollout method
    def ppo_planner_wrapper(img):
        # Convert image to tensor
        tensor_img = torch.FloatTensor(img)
        # Get action from policy (deterministic)
        with torch.no_grad():
            action, _, _, _ = planner.get_action(tensor_img, deterministic=True)
        # Return numpy action
        return action
    
    for t in args.track:
        print(f"Testing on track: {t}")
        
        # Use PyTux's rollout method
        steps, how_far = pytux.rollout(
            t, 
            control, 
            planner=ppo_planner_wrapper,
            max_frames=args.max_frames, 
            verbose=args.verbose
        )
        
        results[t] = {'steps': steps, 'how_far': how_far}
        print(f"Track {t}: Steps = {steps}, Progress = {how_far:.2f}")
    
    # Print summary
    print("\nTest Results Summary:")
    print("=====================")
    for track, result in results.items():
        print(f"Track: {track}")
        print(f"  Steps: {result['steps']}")
        print(f"  Progress: {result['how_far']:.2f}")
    
    pytux.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test the PPO planner")
    parser.add_argument('track', nargs='+', default=['zengarden', 'lighthouse', 'hacienda', 'snowtuxpeak', 'cornfield_crossing', 'scotland'])
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--max_frames', type=int, default=1000)
    parser.add_argument('-f', '--file', type=str, default='ppo_planner.pth')
    
    args = parser.parse_args()
    test_planner(args)