import numpy as np

from pre_train import pretrain_agent, PPO, ExpertDataSet, evaluate_policy, random_split

data = np.load('expert_data.npz')
expert_actions = data['expert_actions']
expert_observations =data['expert_observations']

expert_dataset = ExpertDataSet(expert_observations, expert_actions)

train_size = int(0.6 * len(expert_dataset))

test_size = len(expert_dataset) - train_size

train_expert_dataset, test_expert_dataset = random_split(
    expert_dataset, [train_size, test_size]
)

print("test_expert_dataset: ", len(test_expert_dataset))
print("train_expert_dataset: ", len(train_expert_dataset))


from new_models.walker2d import Walker2dEnv
new_env = Walker2dEnv()



# add new model
new_model = PPO('MlpPolicy', new_env)
print("new_env: ", new_env.reset().shape)


new_model = pretrain_agent(new_model, new_env, train_expert_dataset, test_expert_dataset)

new_model.save("walker2d_pretrain.pkl")

mean_reward, std_reward = evaluate_policy(new_model, new_env, n_eval_episodes=1000)
print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")