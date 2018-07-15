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

    def get_parameters(self):
        # Return flat (1D) model parameters
        # Bug/feature with ray? - doesn't work with pytorch variables

        # detach to remove autograd and allow numpy conversion
        parameters = parameters_to_vector(self.model.parameters()).detach()

        return parameters.numpy()

    def set_parameters(self, parameters, perturbation, evaluate=False):
        # set parameters from sampled deltas
        perturb_parameters = torch.from_numpy(
            parameters + (self.sigma * perturbation))

        vector_to_parameters(perturb_parameters, self.model.parameters())

    def evaluate(self, state):
        prediction = self.model(state)
        action = np.argmax(prediction.data.numpy())

        return action

    def update(self, theta, all_rewards, population):
        # add 1e-5 to prevent division by zero
        normalized_rewards = (np.asarray(all_rewards) -
                              np.mean(all_rewards)) / (np.std(all_rewards)+1e-5)

        new_theta = theta + self.learning_rate / torch.from_numpy(
            (len(population) * self.sigma) * np.dot(np.asarray(population).T, normalized_rewards)).float()

        # set new parameters to the model, for later retrival
        vector_to_parameters(new_theta, self.model.parameters())

    def save_model(self, output_path):
        torch.save(self.model.state_dict(), output_path)
