# Code is run from the python package rl-baselines3-zoo

# import gymnasium as gym

# from stable_baselines3 import PPO
# from stable_baselines3.common.env_util import make_vec_env


# # env_id = "CartPole-v1"
# # env_id = "ALE/SpaceInvaders-v5"
# env_id = "SpaceInvadersNoFrameskip-v4"


# # Parallel environments (n_envs)
# vec_env = make_vec_env(env_id=env_id, n_envs=4)

# model = PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=25000)
# model.save("ppo_cartpole")

# del model # remove to demonstrate saving and loading

# model = PPO.load("ppo_cartpole")

# obs = vec_env.reset()
# while True:
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render("human")