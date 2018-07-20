import ray
import gym
import torch
from torchvision import transforms
import numpy as np

from policy import Policy


@ray.remote
def create_shared_noise():
    """Create a large array of noise to be shared by all workers."""
    noise = np.random.RandomState(123).randn(250000000).astype(np.float32)
    return noise


class SharedNoiseTable(object):
    def __init__(self, noise):
        self.noise = noise
        assert self.noise.dtype == np.float32

    def get(self, index, dim):
        return self.noise[index:index + dim]

    def sample_index(self, dim):
        return np.random.randint(0, len(self.noise) - dim + 1)


@ray.remote
class Worker(object):

    def __init__(self, policy_params, env_name, noise):
        self.env = gym.make(env_name)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((128, 128)),
            # transforms.Grayscale(),
            transforms.ToTensor()
        ])
        self.noise = SharedNoiseTable(noise)
        self.policy = Policy(**policy_params)

    def do_rollouts(self, parameters, render=False):
        # Perform simulation and return reward
        state = self.env.reset()
        done = False
        rollout_reward = 0

        while not done:
            if render:
                self.env.render()

            noise_index = self.noise.sample_index(self.policy.num_params)
            perturbation = self.noise.get(
                noise_index, self.policy.num_params)

            # input perturbation and jitters
            # the model ends up beign only used for foward passes during rollout
            self.policy.set_parameters(parameters, perturbation)

            state = self.transform(state).unsqueeze(0)

            # Do rollout with the perturbed policy.
            action = self.policy.evaluate(state)

            state, reward, done, _ = self.env.step(action)

            rollout_reward += reward

        # Return the rewards.
        return {"noise_index": noise_index, "rollout_reward": rollout_reward}


# Stratey
batch_size = 32  # aka population size
policy_params = {
    "sigma": 0.1,
    "learning_rate": 0.001
}

# Model
num_features = 8

# Distributed
num_workers = 8

# Training
steps = 1000

env_name = "SpaceInvaders-v0"

env = gym.make(env_name)
action_space = env.action_space.n

policy_params["action_space"] = action_space

ray.init()

noise_id = create_shared_noise.remote()
noise = SharedNoiseTable(ray.get(noise_id))

# Instanciate parent policy
policy = Policy(**policy_params)

# Create the actors/workers
workers = [Worker.remote(policy_params, env_name, noise_id)
           for _ in range(num_workers)]

total_rewards = []

highest_reward = 0
for i in range(steps):

    # Loop to fill batch based on number of workers
    rollout_ids = []
    for j in range(batch_size//num_workers):
        # Get the current policy weights.
        theta = policy.get_parameters()

        # Put the current policy weights in the object store.
        theta_id = ray.put(theta)

        # Use the actors to do rollouts,
        # note that we pass in the ID of the policy weights.
        rollout_ids += [worker.do_rollouts.remote(
            theta_id) for worker in workers]

    # Get the results of the rollouts.
    results = ray.get(rollout_ids)

    # Loop over the results.
    all_rollout_rewards, population = [], []
    for result in results:
        all_rollout_rewards.append(result["rollout_reward"])

        _noise = noise.get(result["noise_index"], policy.num_params)
        population.append(_noise)

    avg_reward = np.average(np.asarray(all_rollout_rewards))

    print("average reward in episode ", i+1, ": ", avg_reward)

    # Update parent parameters
    policy.update(theta, all_rollout_rewards, population)

    # Save highest average reward
    if avg_reward > highest_reward:
        highest_reward = avg_reward
        policy.save_model('./best-model.pth')
        print("saved model at episode: ", i+1)

    print("\n")
