from chain_reaction_env import ChainReactionEnv  # Import the class from earlier
from stable_baselines3 import PPO
from sb3_contrib.ppo_mask import MaskablePPO


model = MaskablePPO.load("ppo_chainreaction")
env = ChainReactionEnv(size=6)
obs, _ = env.reset()
done = False
while not done:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)
    env.render()
    input()
print("Reward:", reward)
