import torch
import torch.utils.tensorboard as tb
import numpy as np
from os import path
import argparse

# Import your components
from ppo_planner import PPOPlanner, save_model, PPOPlannerWithLayerNorm
from ppo_agent import PPO
from reward import calculate_reward
from ppo_utils import compute_gae, log_visualization, video_from_tensor
from controller import control
from ppo_pytux_adapter import PPODataCollector


def train(args):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not torch.cuda.is_available() and torch.backends.mps.is_available():
        device = torch.device('mps')
    print(f"Using device: {device}")

    # Set detect anomaly for debugging
    torch.autograd.set_detect_anomaly(True)
    
    # Initialize model
    policy = PPOPlanner().to(device)
    def set_bn_eval(module):
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
    policy.apply(set_bn_eval)

    if args.continue_training:
        try:
            policy.load_state_dict(torch.load(path.join(path.dirname(path.abspath(__file__)), 'ppo_planner.th'), 
                                             map_location=device))
            print("Loaded existing model for continued training")
        except:
            print("No existing model found, starting fresh")
    
    # Initialize optimizer
    # optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    optimizer = torch.optim.Adam(policy.parameters(), lr=args.learning_rate)
    
    # Initialize PPO agent
    ppo_agent = PPO(
        policy=policy,
        optimizer=optimizer,
        clip_param=args.clip_param,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        target_kl_heuristic=args.target_kl,
        device=device
    )
    
    # Create logger
    train_logger = None
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train_ppo'))
    
    # Initialize data collector
    
    total_steps = 0
    best_reward = -float('inf')
    data_collector = PPODataCollector(
        policy=policy,
        calculate_reward_fn=calculate_reward,
        control_fn=control,
        device=device
        )
    
    # Training loop
    for epoch in range(args.num_epoch):
        epoch_rewards = []
        epoch_lengths = []
        epoch_progresses = []
        
        for track in args.tracks:
            print(f"Epoch {epoch+1}/{args.num_epoch}, Track: {track}")
            
            # Collect episode data
            rollout = data_collector.collect_episode(
                track=track,
                max_frames=args.max_steps,
                verbose=args.verbose
            )
            
            if len(rollout['rewards']) == 0:
                print(f"No steps collected for track {track}, skipping update")
                continue
                
            epoch_rewards.append(rollout['episode_reward'])
            epoch_lengths.append(rollout['episode_length'])
            epoch_progresses.append(rollout['final_progress'])
            
            # Compute returns and advantages
            returns, advantages = compute_gae(
                rewards=rollout['rewards'],
                values=rollout['values'],
                dones=rollout['dones'],
                last_value=rollout['last_value'],
                gamma=args.gamma,
                gae_lambda=args.gae_lambda
            )
            
            # Update policy
            update_info = ppo_agent.update(
                obs=rollout['obs'],
                actions=rollout['actions'],
                old_log_probs=rollout['log_probs'],
                returns=returns,
                advantages=advantages,
                update_epochs=args.update_epochs,
                batch_size=args.batch_size
            )
            
            # Log statistics
            if train_logger is not None:
                train_logger.add_scalar(f'reward/{track}', rollout['episode_reward'], total_steps)
                train_logger.add_scalar(f'episode_length/{track}', rollout['episode_length'], total_steps)
                train_logger.add_scalar(f'progress/{track}', rollout['final_progress'], total_steps)
                train_logger.add_scalar('policy_loss', update_info['policy_loss'], total_steps)
                train_logger.add_scalar('value_loss', update_info['value_loss'], total_steps)
                train_logger.add_scalar('entropy', update_info['entropy'], total_steps)
                train_logger.add_scalar('kl', update_info['kl'], total_steps)
                train_logger.add_scalar('rewards/mean', np.mean(rollout['rewards']), total_steps)
                train_logger.add_scalar('rewards/max', np.max(rollout['rewards']), total_steps)
                train_logger.add_scalar('rewards/min', np.min(rollout['rewards']), total_steps)
                train_logger.add_scalar('values/mean', np.mean(rollout['values']), total_steps)
                # Log gradient norms
                grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), args.max_grad_norm, norm_type=2)
                train_logger.add_scalar('training/grad_norm', grad_norm, total_steps)
                # Visualize a random observation and prediction

                if len(rollout['obs']) > 0:
                    idx = np.random.randint(len(rollout['obs']))
                    sample_obs = torch.FloatTensor(rollout['obs'][idx:idx+1]).to(device)
                    sample_action_indices = torch.FloatTensor(rollout['actions'][idx:idx+1]).to(device)
                    sample_action = torch.tensor([(sample_action_indices % 16) / 8 - 1, (sample_action_indices // 16) / 6 - 1], dtype=torch.float32).to(device)
                    with torch.no_grad():
                        pred_action, _, _, _ = policy.get_action(sample_obs, deterministic=True)
                    log_visualization(train_logger, sample_obs, sample_action, pred_action, total_steps)
            
            print(f"  Reward: {rollout['episode_reward']:.2f}, Steps: {rollout['episode_length']}, Progress: {rollout['final_progress']:.2f}")
            print(f"  Policy Loss: {update_info['policy_loss']:.4f}, Value Loss: {update_info['value_loss']:.4f}")
            
            total_steps += rollout['episode_length']
        
        # Calculate epoch statistics
        if epoch_rewards:
            avg_reward = sum(epoch_rewards) / len(epoch_rewards)
            avg_progress = sum(epoch_progresses) / len(epoch_progresses)
            
            print(f"Epoch {epoch+1} summary:")
            print(f"  Average Reward: {avg_reward:.2f}")
            print(f"  Average Progress: {avg_progress:.2f}")
            
            # Save best model
            if avg_reward > best_reward:
                best_reward = avg_reward
                save_model(policy)
                print(f"  New best model saved! Reward: {best_reward:.2f}")
        
        # Save model periodically
        if (epoch + 1) % args.save_interval == 0:
            save_path = path.join(path.dirname(path.abspath(__file__)), f'ppo_planner_epoch_{epoch+1}.th')
            # video_from_tensor(rollout['obs'], path.join(path.dirname(path.abspath(__file__)), f'video_epoch_{epoch+1}.mp4'), fps=10)
            torch.save(policy.state_dict(), save_path)
            print(f"  Model checkpoint saved at epoch {epoch+1}")
    
    # Save final model
    save_model(policy)
    data_collector.close()
    if train_logger is not None:
        train_logger.close()
    print("Training completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--tracks', nargs='+', default=['zengarden', 'lighthouse', 'hacienda', 'snowtuxpeak', 'cornfield_crossing', 'scotland', "cocoa_temple"])
    parser.add_argument('-n', '--num_epoch', type=int, default=50)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4)
    parser.add_argument('--clip_param', type=float, default=0.1)
    parser.add_argument('--value_coef', type=float, default=0.5)
    parser.add_argument('--entropy_coef', type=float, default=0.01)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.90)
    parser.add_argument('--update_epochs', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)
    parser.add_argument('--target_kl', type=float, default=0.01)
    parser.add_argument('--save_interval', type=int, default=25)
    parser.add_argument('-c', '--continue_training', action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    
    args = parser.parse_args()
    train(args)