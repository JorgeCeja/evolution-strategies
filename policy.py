import torch
import copy
import numpy as np
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from model import Model


class Policy(object):

    def __init__(self, sigma=0.03, learning_rate=0.001, action_space=None):
        self.model = Model(action_space)
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.num_params = self.model.count_parameters()

        # Remove grad from model
        for param in self.model.parameters():
            param.requires_grad = False

    def get_parameters(self):
        # Return flat (1D) model parameters
        # Bug/feature with ray? - doesn't work with pytorch variables

        # Don't need to detach to allow numpy conversion
        # grad was removed from model in init
        parameters = parameters_to_vector(self.model.parameters())

        return parameters.numpy()

    def set_parameters(self, parameters, perturbation, evaluate=False):
        # Set parameters from sampled deltas
        perturb_parameters = torch.from_numpy(
            parameters + (self.sigma * perturbation))

        vector_to_parameters(perturb_parameters, self.model.parameters())

    def evaluate(self, state):
        # No need for `with torch.no_grad():`
        # grad was removed from model in init
        prediction = self.model(state)
        action = np.argmax(prediction.data.numpy())

        return action

    def update(self, theta, all_rewards, population):
        # Clip denominator to 1e-5 to prevent division by 0
        normalized_rewards = (np.asarray(all_rewards) -
                              np.mean(all_rewards)) / (np.clip(np.std(all_rewards), 1e-5, None))

        new_weights = theta + self.learning_rate / torch.from_numpy(
            (len(population) * self.sigma) * np.dot(np.asarray(population).T, normalized_rewards)).float()

        # set new parameters to the model, for later retrival
        vector_to_parameters(new_weights, self.model.parameters())

    def save_model(self, output_path):
        torch.save(self.model.state_dict(), output_path)
