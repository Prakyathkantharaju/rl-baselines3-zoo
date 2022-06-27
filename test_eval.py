import numpy as np
import os
import pandas as pd
from stable_baselines3 import PPO, A2C, TD3, SAC
from torch import seed
import gym

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy




states = None
for j in range(10):
	new_env = gym.make('Walker2d-v3')
	new_env = Monitor(new_env)
	obs  = new_env.reset()
	new_model = PPO('MlpPolicy', new_env)
	new_model.load("logs/new_opt/trial_35/best_model.zip")
	mean_reward, std_reward = evaluate_policy(new_model, new_env, n_eval_episodes=10)
	print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
	reward_sum = 0
	reward_threshold = 900

	frame_store = []
	if mean_reward > reward_threshold:
		print("Found good model")







for root, dir, files in os.walk('logs/new_opt/'):
	if len(files) > 0:
		if files[1].endswith('.npz'):
			model_path = root + '/' +files[1]
			a = np.load(model_path)
			d = dict(zip(("data1{}".format(k) for k in a), (a[k] for k in a)))
			os.makedirs('walker_hyper_eval_results/' + root.split('/')[-1], exist_ok=True)
			df_results = pd.DataFrame(d['data1results'])
			df_results.index = d['data1timesteps']
			df_results.to_csv('walker_hyper_eval_results/' + root.split('/')[-1] + '/results.csv')
			df_hyper = pd.DataFrame(d['data1ep_lengths'])
			df_hyper.index = d['data1timesteps']
			df_hyper.to_csv('walker_hyper_eval_results/' + root.split('/')[-1] + '/hyperparameters.csv')
			try:
				# print(root.split('/')[-1], max(max(d['data1results'])), max(max(d['data1ep_lengths'])))
				# print(d['data1results'])
				if any(d['data1results'] > 500):
					print('########### Found best model ###########')
			except ValueError:
				pass
			states = None
		if files[0].endswith('.zip'):
			new_env = gym.make('Walker2d-v3')
			new_env = Monitor(new_env)
			obs  = new_env.reset()
			new_model = PPO('MlpPolicy', new_env)
			new_model.load(root + '/' + files[0])
			mean_reward, std_reward = evaluate_policy(new_model, new_env, n_eval_episodes=10)
			print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
			reward_sum = 0
			reward_threshold = 900

			frame_store = []
			if mean_reward > reward_threshold:
				print("Found good model")
			
			