
from stable_baselines3 import PPO, A2C, TD3, SAC
from new_models.walker2d import Walker2dEnv


new_env = Walker2dEnv()
obs  = new_env.reset()


new_model = PPO('MlpPolicy', new_env)
print("new_env: ", new_env.reset().shape)
new_model.load("walker2d_pretrain.pkl")


for i in range(1000):
	action, _ = new_model.predict(obs)
	obs, rewards, done, info = new_env.step(action)
	new_env.render()
	print(rewards)
	if done:
		obs = new_env.reset()
		print("done")
		break