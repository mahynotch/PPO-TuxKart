import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PPO:
    def __init__(self, policy, optimizer, clip_param=0.2, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5, target_kl_heuristic=0.015, device='cpu'):
        self.policy = policy
        self.optimizer = optimizer
        self.clip_param = clip_param
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl_heuristic = target_kl_heuristic
        self.device = device

    def update(self, obs, actions, old_log_probs, returns, advantages, update_epochs=4, batch_size=64):
        # Convert to tensors
        obs_tensor = torch.FloatTensor(np.array(obs)).to(self.device)
        actions_tensor = torch.LongTensor(actions).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        returns_tensor = torch.FloatTensor(returns).to(self.device)
        advantages_tensor = torch.FloatTensor(advantages).to(self.device)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        dataset = torch.utils.data.TensorDataset(obs_tensor, actions_tensor, old_log_probs_tensor, returns_tensor, advantages_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        total_kl_heuristic = 0
        updates_performed_count = 0 # Initialize counter for actual updates

        early_stop_triggered = False
        for epoch in range(update_epochs):
            if early_stop_triggered:
                break # Stop processing more epochs if early stop was triggered in a previous one

            # print("dataloader length:", len(dataloader))

            for batch_obs, batch_actions, batch_old_log_probs, batch_returns, batch_advantages in dataloader:
                heatmap_logits, value = self.policy(batch_obs)
                current_batch_size = heatmap_logits.size(0)
                flat_heatmap_logits = heatmap_logits.view(current_batch_size, -1)
                dist = torch.distributions.Categorical(logits=flat_heatmap_logits)

                entropy = dist.entropy().mean()

                log_probs = dist.log_prob(batch_actions.squeeze(-1) if batch_actions.dim() > 1 else batch_actions)
                ratio = torch.exp(log_probs - (batch_old_log_probs.squeeze(-1) if batch_old_log_probs.dim() > 1 else batch_old_log_probs))
                adv_squeezed = batch_advantages.squeeze(-1) if batch_advantages.dim() > 1 else batch_advantages
                surr1 = -ratio * adv_squeezed
                surr2 = -torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_squeezed
                policy_loss = torch.max(surr1, surr2).mean()

                value_squeezed = value.squeeze(-1)
                batch_returns_squeezed = batch_returns.squeeze(-1) if batch_returns.dim() > 1 else batch_returns     
                value_loss = F.mse_loss(value_squeezed, batch_returns_squeezed)

                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                log_ratio = log_probs - (batch_old_log_probs.squeeze(-1) if batch_old_log_probs.dim() > 1 else batch_old_log_probs)
                approx_kl = 0.5 * torch.mean(log_ratio.pow(2)).item()
                # approx_kl = torch.mean(log_ratio.exp()).item()

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                updates_performed_count += 1 # Increment after each successful step

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                total_kl_heuristic += approx_kl

                if self.target_kl_heuristic is not None and approx_kl > self.target_kl_heuristic:
                    # print(f"Epoch {epoch+1}, Batch: Early stopping due to KL divergence {approx_kl:.4f} > {self.target_kl_heuristic:.4f}")
                    print(f"Early stopping due to KL divergence {approx_kl:.4f} > {self.target_kl_heuristic:.4f}, time updated: {updates_performed_count}")
                    early_stop_triggered = True
                    break # Break from inner loop (batches)
        
        # Calculate averages using the actual number of updates performed
        if updates_performed_count == 0: # Should not happen if dataloader is not empty
            avg_policy_loss = 0
            avg_value_loss = 0
            avg_entropy = 0
            avg_kl_heuristic = 0
        else:
            avg_policy_loss = total_policy_loss / updates_performed_count
            avg_value_loss = total_value_loss / updates_performed_count
            avg_entropy = total_entropy / updates_performed_count
            avg_kl_heuristic = total_kl_heuristic / updates_performed_count
        
        return {
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'kl': avg_kl_heuristic,
            'updates_performed': updates_performed_count # Now this is correctly tracked
        }