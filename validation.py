
from stable_baselines3 import PPO, A2C, TD3, SAC
from torch import seed
from new_models.walker2d import Walker2dEnv

from mujoco_py import GlfwContext
GlfwContext(offscreen=True)

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy


new_env = Walker2dEnv()
new_env.seed(0)
new_env = Monitor(new_env)
obs  = new_env.reset()


new_model = PPO('MlpPolicy', new_env)
print("new_env: ", new_env.reset().shape)
new_model.load("logs/ppo/Walker2d-v3_2/best_model.zip")
mean_reward, std_reward = evaluate_policy(new_model, new_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

new_model.load("logs/trial_3/best_model.zip")
mean_reward, std_reward = evaluate_policy(new_model, new_env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")


# gym to gif: https://gist.github.com/botforge/64cbb71780e6208172bbf03cd9293553
from matplotlib import animation
import matplotlib.pyplot as plt
import gym 

"""
Ensure you have imagemagick installed with 
sudo apt-get install imagemagick
Open file in CLI with:
xgd-open <filelname>
"""
def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


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
	# if True:
		for i in range(1000):
			action, states = new_model.predict(obs, state=states, deterministic=True)
			obs, rewards, done, info = new_env.step(action)

			frame_store.append(new_env.render("rgb_array"))
			# print(rewards)
			reward_sum += rewards
			if done:
				obs = new_env.reset()
				print("done", reward_sum)
				print("reward_sum: ", reward_sum)
				save_frames_as_gif(frame_store, filename='walker2d_pretrain_trial_4_' + str(j) + '.gif')
				break

"""
{'batch_size': 512, 'n_steps': 1024, 'gamma': 0.9, 'learning_rate': 1.841645507088866e-05, 'ent_coef': 0.00011040674836880027, 'clip_range': 0.4, 'n_epochs': 10, 'gae_lambda': 1.0, 'max_grad_norm': 2, 'vf_coef': 0.269032939640485, 'net_arch': 'medium', 'activation_fn': 'tanh'}.
{'batch_size': 256, 'n_steps': 64, 'gamma': 0.95, 'learning_rate': 0.016253652447000624, 'ent_coef': 4.7102691368895975e-05, 'clip_range': 0.2, 'n_epochs': 10, 'gae_lambda': 0.8, 'max_grad_norm': 0.6, 'vf_coef': 0.9961553137120291, 'net_arch': 'medium', 'activation_fn': 'tanh'}.
{'batch_size': 128, 'n_steps': 32, 'gamma': 0.9999, 'learning_rate': 0.012577316380371906, 'ent_coef': 1.0551506810586044e-07, 'clip_range': 0.4, 'n_epochs': 10, 'gae_lambda': 0.98, 'max_grad_norm': 0.3, 'vf_coef': 0.26077553268637965, 'net_arch': 'small', 'activation_fn': 'tanh'} -> 900 +,


    batch_size: 256
    n_steps: 512
    gamma: 0.995
    learning_rate: 0.00016152763919747857
    ent_coef: 4.543132151457905e-05
    clip_range: 0.1
    n_epochs: 1
    gae_lambda: 0.8
    max_grad_norm: 5
    vf_coef: 0.12664205796160521
    net_arch: medium
    activation_fn: relu

"""