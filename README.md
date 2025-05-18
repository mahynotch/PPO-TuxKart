# PPO-TuxKart
PPO-TuxKart is a reinforcement learning agent that uses Proximal Policy Optimization (PPO) to play the TuxKart game. To install the required dependencies, run the following command:
```bash
conda create -n tuxkart python=3.9
conda activate tuxkart
pip install -U PySuperTuxKart
pip install matplotlib
pip install torch torchvision tensorboard --index-url https://download.pytorch.org/whl/cu126
```

## Training the Agent
To train the agent, run the following command:
```bash
python train.py -n <num_epoch> -lr <learning_rate> --max_steps <max_steps> --tracks <zengarden, lighthouse, ...>
```
For more options please check the `ppo_train.py` file, you can add `-v` at the end for visualizing.

## Testing the Agent
To test the agent, run the following command:
```bash
python test.py <zengarden, lighthouse, ...> -f <model_weight_path>
```
For more options please check the `ppo_test.py` file, you can add `-v` at the end for visualizing. Some trained models are available in the "weights" folder. 