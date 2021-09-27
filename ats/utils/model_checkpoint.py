import os
import numpy as np
import torch

def save_checkpoint(model, optimizer, save_path, epoch):
    state_dict = {}
    state_dict['model_state_dict'] = model.state_dict()
    state_dict['optimizer_state_dict'] = optimizer.state_dict()
    state_dict['epoch'] = epoch
    torch.save(state_dict, save_path)


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


class ModelCheckpoint:
    def __init__(self, output_directory, ats_model, optimizer, save_best=True, save_frequency=False, lower_better=True):
        self.output_directory = output_directory
        os.makedirs(output_directory, exist_ok=True)
        self.save_frequency = save_frequency
        self.save_best = save_best
        self.lower_better = lower_better

        self.ats_model = ats_model
        self.optimizer = optimizer

        self.best_metric = np.finfo(np.float64).max if lower_better else np.finfo(np.float64).min

    def __call__(self, epoch, metric):
        if metric < self.best_metric:
            self.best_metric = metric
            if self.save_best:
                self._save_model("model_best.pth", epoch)

        if self.save_frequency and epoch % self.save_frequency == 0:
            self._save_model(f"model_{epoch}.pth", epoch)

    def _save_model(self, name, epoch):
        save_checkpoint(self.ats_model, self.optimizer, os.path.join(self.output_directory, name), epoch)
