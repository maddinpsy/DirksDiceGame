from stable_baselines3 import PPO
from chain_reaction_env import ChainReactionEnv  # Import the class from earlier
from stable_baselines3.common.env_util import make_vec_env
from small_cnn import SmallCNN
from gymnasium.wrappers import TimeLimit
    
policy_kwargs = dict(
    features_extractor_class=SmallCNN,
    features_extractor_kwargs=dict(features_dim=64),
)


# Create environment
env = make_vec_env(
    lambda: TimeLimit(ChainReactionEnv(size=6), max_episode_steps=20),
    n_envs=8
)

# Train PPO
# model = PPO.load("ppo_chainreaction", env=env)
model = PPO("MlpPolicy", env,  verbose=1, tensorboard_log="./tensorboard_logs/")
# model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs,  verbose=1, tensorboard_log="./tensorboard_logs/")
model.learn(total_timesteps=500_000)

# Save
model.save("ppo_chainreaction")

# run in second terminal
#tensorboard --logdir tensorboard_logs --host 0.0.0.0 --port 6006
# access tensor board at:
# https://gdasn4cpk6gewh9.studio.us-east-2.sagemaker.aws/studiolab/default/jupyter/proxy/6006/?darkMode=true#timeseries