from stable_baselines3 import PPO
from sb3_contrib.ppo_mask import MaskablePPO
from chain_reaction_env import ChainReactionEnv  # Import the class from earlier
from stable_baselines3.common.env_util import make_vec_env
from small_cnn import SmallCNN
    
policy_kwargs = dict(
    features_extractor_class=SmallCNN,
    features_extractor_kwargs=dict(features_dim=64),
)


# Create environment
env = make_vec_env(lambda: ChainReactionEnv(size=6), n_envs=8)

# Train PPO
model = MaskablePPO("CnnPolicy", env, policy_kwargs=policy_kwargs,  verbose=1, tensorboard_log="./tensorboard_logs/")
model.learn(total_timesteps=500_000)

# Save
model.save("ppo_chainreaction")

# run in second terminal
#tensorboard --logdir tensorboard_logs --host 0.0.0.0 --port 6006
# access tensor board at:
# https://gdasn4cpk6gewh9.studio.us-east-2.sagemaker.aws/studiolab/default/jupyter/proxy/6006/?darkMode=true#timeseries