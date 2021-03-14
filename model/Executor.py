from . import convolutional_neural_network
import torch
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
from tqdm.notebook import tqdm
import numpy as np


class Executor:
    def __init__(self):
        self.network = convolutional_neural_network.convolutional_neural_network()
        self.entropy_loss = nn.CrossEntropyLoss()
        self.gd_reducer = optim.SGD(self.network.parameters(), lr=0.01, momentum=0.9)

    def load_dataset(self):
        # set the dataset part of preprocessing
        # self.train_data = None
        pass

    # To avoid the overflow for a given number;
    # by normalising it with some probability value
    def assign_images_with_probability_values(self, images):
        result = self.network(images)
        index_values = (torch.max(result, 1)[1]).numpy()
        # Squeeze by ignoring 1D arrays
        updated_index_values = np.squeeze(index_values)
        probability_values = []
        for idx, model_value in zip(updated_index_values, result):
            calculated_sf_max = functional.softmax(model_value, dim=0)
            probability_value = calculated_sf_max[idx].item()
            probability_values.append(probability_value)
        return index_values, probability_values

    def calculate_model_accuracy(self, data_src):
        accuracy = 0
        all_results = 0
        with torch.no_grad():
            self.network.eval()
            for d in data_src:
                op = self.assign_images_with_probability_values(d[0])
                all_results = all_results + len(d[1].numpy())
                accuracy = accuracy + (op[0] == d[1].numpy()).sum()

        acc_in_percent = (accuracy / all_results) * 100
        return acc_in_percent

    def training_model_executor(self, data_src, num_iters):
        all_loss_vals = []
        each_iter_loss = []
        calc_min_loss = float("inf")
        all_accuracy_vals = []
        # Track loading using tqdm
        for _ in tqdm(range(0, num_iters), total=num_iters, desc="train model"):
            self.network.train()
            for item in data_src:
                # 0:images 1:labels
                result = self.network(item[0])
                calc_loss = self.entropy_loss(result, item[1])
                all_loss_vals.append(calc_loss.item())
                self.gd_reducer.zero_grad()
                calc_loss.backward()
                self.gd_reducer.step()
            total_iter_loss = sum(all_loss_vals)
            covered_so_far = len(all_loss_vals)
            avg_loss = total_iter_loss / covered_so_far
            each_iter_loss.append(avg_loss)
            calc_accuracy = self.calculate_model_accuracy(data_src)
            all_accuracy_vals.append(calc_accuracy)
            print("current iter acc:", calc_accuracy)
            print("current iter loss:", calc_min_loss)
            calc_min_loss = min(avg_loss, calc_min_loss)

        return all_loss_vals, each_iter_loss, all_accuracy_vals

# if __name__ == '__main__':
