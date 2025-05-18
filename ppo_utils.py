import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


def collect_rollouts(pytux, policy, track, calculate_reward_fn, control_fn, max_steps=1000):
    """Collect experiences from the environment"""
    device = next(policy.parameters()).device
    
    obs_list = []
    action_list = []
    log_prob_list = []
    reward_list = []
    done_list = []
    value_list = []
    
    # Initialize episode
    obs = pytux.reset(track)
    done = False
    episode_reward = 0
    step = 0
    
    last_progress = 0
    
    while not done and step < max_steps:
        # Convert observation to tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        
        # Get action from policy
        with torch.no_grad():
            action, log_prob, value = policy.get_action(obs_tensor)
        
        # Convert to numpy
        action_np = action.cpu().numpy()[0]
        
        # Execute action in environment using the controller
        current_vel = pytux.get_velocity()
        control_action = control_fn(action_np, current_vel)
        
        # Step the environment
        next_obs, done, info = pytux.step(control_action)
        
        # Calculate progress difference
        current_progress = info.get('progress', 0)
        progress_diff = current_progress - last_progress
        last_progress = current_progress
        
        # Calculate reward
        reward = calculate_reward_fn(
            observation=obs,
            action=control_action,
            done=done,
            progress=progress_diff,
            velocity=current_vel
        )
        
        # Store experience
        obs_list.append(obs)
        action_list.append(action_np)
        log_prob_list.append(log_prob.cpu().numpy()[0])
        reward_list.append(reward)
        done_list.append(done)
        value_list.append(value.cpu().numpy()[0])
        
        # Update for next iteration
        obs = next_obs
        episode_reward += reward
        step += 1
    
    # Add final state value for GAE calculation
    if len(obs_list) > 0:
        with torch.no_grad():
            last_obs = torch.FloatTensor(obs).unsqueeze(0).to(device)
            _, last_value = policy(last_obs)
            last_value = last_value.cpu().numpy()[0]
    else:
        last_value = 0
    
    return {
        'obs': np.array(obs_list) if len(obs_list) > 0 else np.array([]),
        'actions': np.array(action_list) if len(action_list) > 0 else np.array([]),
        'log_probs': np.array(log_prob_list) if len(log_prob_list) > 0 else np.array([]),
        'rewards': np.array(reward_list) if len(reward_list) > 0 else np.array([]),
        'dones': np.array(done_list) if len(done_list) > 0 else np.array([]),
        'values': np.array(value_list) if len(value_list) > 0 else np.array([]),
        'last_value': last_value,
        'episode_reward': episode_reward,
        'episode_length': step,
        'final_progress': last_progress
    }


def compute_gae(rewards, values, dones, last_value, gamma=0.99, gae_lambda=0.95):
    """Compute Generalized Advantage Estimation"""
    advantages = np.zeros_like(rewards)
    last_gae = 0
    
    if values.ndim > 1 and values.shape[1] == 1:
        values_1d = values.squeeze(-1)
    elif values.ndim == 1:
        values_1d = values
    else:
        raise ValueError(f"Unexpected shape for 'values' in compute_gae: {values.shape}. Expected (T,) or (T,1).")

    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = last_value
        else:
            next_value = values_1d[t + 1]

        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values_1d[t]
        last_gae = delta + gamma * gae_lambda * (1 - dones[t]) * last_gae
        advantages[t] = last_gae
    returns = advantages + values_1d 
    
    return returns, advantages


def log_visualization(logger, img, label, pred, global_step):
    """Log visualization of predicted vs. actual aim points"""
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(TF.to_pil_image(img[0].cpu()))
    WH2 = np.array([img.size(-1), img.size(-2)])/2
    
    # Get coordinates for label and prediction
    label_coord = WH2*(label[0].cpu().detach().numpy()+1)
    pred_coord = WH2*(pred[0].cpu().detach().numpy()+1)
    
    # Improved visibility with larger circles and better colors
    ax.add_artist(plt.Circle(label_coord, 4, ec='lime', fill=False, lw=2, label='Ground Truth'))
    ax.add_artist(plt.Circle(pred_coord, 4, ec='red', fill=False, lw=2, label='Prediction'))
    
    # Add legend for clarity
    ax.legend(loc='upper right')
    
    logger.add_figure('viz', fig, global_step)
    # plt.savefig(f'viz_{global_step:04d}.png', dpi=100)
    plt.close(fig)

def video_from_tensor(tensor_frames, output_path, fps=30):
    """Create a video from a tensor of frames using PIL."""
    tensor_frames = torch.tensor(tensor_frames)
    try:
        from PIL import Image
        import numpy as np
        import os
        
        # Create temp directory for frames
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        # Save each frame as an image
        frame_files = []
        for i in range(tensor_frames.shape[0]):
            # Convert from tensor to numpy array, scale to 0-255
            frame = tensor_frames[i].permute(1, 2, 0).cpu().numpy()
            frame = (frame * 255).astype(np.uint8)
            
            # Save as image
            img = Image.fromarray(frame)
            frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
            img.save(frame_path)
            frame_files.append(frame_path)
        
        # Use ffmpeg to combine frames (if available)
        try:
            import subprocess
            cmd = [
                'ffmpeg', '-y', '-framerate', str(fps),
                '-i', os.path.join(temp_dir, 'frame_%04d.png'),
                '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
                output_path
            ]
            subprocess.run(cmd, check=True)
            print(f"Video saved to {output_path}")
        except (ImportError, subprocess.CalledProcessError):
            print(f"Warning: Could not create video. Frames saved to {temp_dir}")
    
    except ImportError:
        print(f"Warning: Could not import PIL. Video will not be saved.")
    except Exception as e:
        print(f"Error creating video: {e}")