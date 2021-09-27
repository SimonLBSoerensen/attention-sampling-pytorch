import os

import cv2
import numpy as np
import random
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision
import matplotlib.pyplot as plt
import wandb
from tensorboardX import SummaryWriter
from PIL import Image


class FileWriter:
    def __init__(self, output_directory):
        os.makedirs(output_directory, exist_ok=True)
        self.outout_file = os.path.join(output_directory, "hist.csv")
        self.file_log_header_written = False

    def __call__(self, epoch, losses=None, metrics=None):
        if losses is not None:
            with open(self.outout_file, "a") as f:
                # If a header has not been written write a header for the file
                if not self.file_log_header_written:
                    header_list = ["epoch"]

                    if losses is not None:
                        if len(losses) == 2:
                            header_list += ["train_loss", "test_loss"]
                        else:
                            header_list += ["train_loss"]

                    if metrics is not None:
                        if len(metrics) == 2:
                            for key in metrics[0]:
                                header_list += ["train_" + key]
                            for key in metrics[1]:
                                header_list += ["test_" + key]
                        else:
                            for key in metrics[0]:
                                header_list += ["train_" + key]

                    header = ",".join(header_list) + "\n"
                    f.write(header)

                    self.file_log_header_written = True

                values_to_write = [epoch]
                if losses is not None:
                    values_to_write += [*losses]

                if metrics is not None:
                    for d in metrics:
                        for key in d:
                            values_to_write += [d[key]]

                value_list = ",".join([str(el) for el in values_to_write])
                f.write(value_list + "\n")


class WandBLogger:
    def __init__(self, **wandb_init):
        wandb.init(**wandb_init)
        self.log_dict = {}

    def __call__(self, epoch, losses=None, metrics=None, images=None):
        self.log_dict = {"epoch": epoch}

        if losses is not None:
            if len(losses) == 1:
                self._add_metric({"loss": losses[0]}, split="train")
            else:
                train_loss, test_loss = losses
                self._add_metric({"loss": train_loss}, split="train")
                self._add_metric({"loss": test_loss}, split="test")

        if metrics is not None:
            if len(metrics) == 1:
                self._add_metric(metrics[0], split="train")
            else:
                train_metrics, test_metrics = metrics
                self._add_metric(train_metrics, split="train")
                self._add_metric(test_metrics, split="test")

        if images is not None:
            self.log_image(epoch, images)

        wandb.log(self.log_dict, step=epoch)

    def log_image(self, epoch, images):
        if images is not None:
            wandb_image_dict = {}
            for log_name in images:
                wandb_image_dict[log_name] = [wandb.Image(image) for image in images[log_name]]
                if len(wandb_image_dict[log_name]) == 1:
                    wandb_image_dict[log_name] = wandb_image_dict[log_name][0]

            wandb.log(wandb_image_dict, step=epoch)

    def _add_metric(self, metric, split):
        for metric_name in metric:
            self.log_dict[split + "_" + metric_name] = metric[metric_name]


class TrainingLogger:
    """
    Save information about the training (losses, metrics, attention maps)
    """

    def __init__(self, output_directory, ats_model, dataset, project, use_wandb=True, device=None,
                 write_tensorboard=True,
                 keep_on_gpu=True, make_images_every=1, nsamples=9, nrow=3, send_images_every=None,
                 save_images_to_disk=True, run_name=None):
        # Check if cuda is available and if no device was given then set device accordingly
        cuda_is_available = torch.cuda.is_available()
        if device is None:
            device = torch.device("cuda") if cuda_is_available else torch.device("cpu")
        self.device = device
        self.save_images_to_disk = save_images_to_disk
        self.make_images_every = make_images_every
        self.dir = output_directory
        os.makedirs(self.dir, exist_ok=True)
        self.write_tensorboard = write_tensorboard
        self.image_out_root = os.path.join(self.dir, "images")
        os.makedirs(self.image_out_root, exist_ok=True)

        self.ats_model = ats_model
        self.nrow = nrow

        # Get the data samples to use for attention images
        idxs = dataset.strided(nsamples)
        data = [dataset[i] for i in idxs]

        self.x_low = torch.stack([d[0] for d in data]).cpu()
        self.x_high = torch.stack([d[1] for d in data]).cpu()
        if keep_on_gpu and cuda_is_available:
            self.x_low = self.x_low.to(self.device)
            self.x_high = self.x_high.to(self.device)

        self.labels = torch.LongTensor([d[2] for d in data]).numpy()

        self.use_wandb = use_wandb
        self.send_images_every = send_images_every
        if self.use_wandb:
            self.wandb_logger = WandBLogger(project=project, name=run_name)

        self.fileWriter = FileWriter(self.dir)

        if self.write_tensorboard:
            self.writer = SummaryWriter(self.dir, flush_secs=5)
        self.on_train_begin()

    def save_images(self, epoch, images):
        if self.save_images_to_disk:
            for image_type in images:
                image_type_folder = os.path.join(self.image_out_root, image_type)
                os.makedirs(image_type_folder, exist_ok=True)

                if len(images[image_type]) == 1:
                    images[image_type][0].save(os.path.join(image_type_folder, f"{epoch}.png"))
                else:
                    for i, image in enumerate(images[image_type]):
                        image.save(os.path.join(image_type_folder, f"{epoch}_{i}.png"))

    def send_images(self, epoch, images):
        if self.send_images_every > 0 and (self.use_wandb and epoch > 0 and epoch % self.send_images_every == 0):
            self.wandb_logger(epoch, None, None, images)

    def tensorboard_and_image(self, name, image, global_step, dataformats):
        if self.write_tensorboard:
            self.writer.add_image(name, image, global_step=global_step, dataformats=dataformats)

    def on_train_begin(self):
        # Save a grid with the images used for att images
        with torch.no_grad():
            _, _, _, x_low, _ = self.ats_model(self.x_low.to(self.device), self.x_high.to(self.device))
            x_low = x_low.cpu()
            image_list = [x for x in x_low]

        grid = torchvision.utils.make_grid(image_list, nrow=self.nrow, normalize=True, scale_each=True)

        self.tensorboard_and_image('original_images', grid, global_step=0, dataformats='CHW')
        images = {
            "original_images": [transforms.ToPILImage()(grid)]
        }
        self.save_images(-1, images)
        self.__call__(-1, return_images=False)

        if self.send_images_every > 0 and self.use_wandb:
            self.wandb_logger.log_image(0, images)

    def __call__(self, epoch, losses=None, metrics=None, return_images=False,
                 use_masked_superimposed=False, color_map=cv2.COLORMAP_JET, interpolate_mode="bicubic"):
        if metrics is not None:
            if len(metrics) == 1:
                self._add_metric(metrics[0], epoch, split="Train")
            else:
                train_metrics, test_metrics = metrics
                self._add_metric(train_metrics, epoch, split="Train")
                self._add_metric(test_metrics, epoch, split="Test")

        if losses is not None:
            if len(losses) == 1:
                self._add_metric({"loss": losses[0]}, epoch, split="Train")
            else:
                train_loss, test_loss = losses
                self._add_metric({"loss": train_loss}, epoch, split="Train")
                self._add_metric({"loss": test_loss}, epoch, split="Test")

        self.fileWriter(epoch, losses, metrics)

        image_epoch = (epoch % self.make_images_every == 0) \
                      and self.make_images_every or epoch < 0

        if image_epoch:
            with torch.no_grad():
                # Get the attention maps, interpolate them to the same size as the
                # low image resolution images and bring them to the CPU
                y, att, patches, x_low, offsets = self.ats_model(self.x_low.to(self.device),
                                                                 self.x_high.to(self.device))
                att = att.unsqueeze(1)
                att = F.interpolate(att, size=(x_low.shape[-2], x_low.shape[-1]), mode=interpolate_mode,
                                    align_corners=False)
                att = att.cpu()

                x_low = x_low.cpu()

            superimposed_list = []
            vision_list = []
            chosen_samples_list = []
            for att_el, x_low_el, offsets_el in zip(att, x_low, offsets):
                # TODO optimize (can be done with matrix opr's)
                x_low_np = x_low_el.permute(1, 2, 0).numpy()
                x_low_np = (x_low_np * 255).astype(np.uint8)

                att_np = att_el.squeeze().detach().numpy()
                att_np_norm = (att_np - att_np.min()) / (att_np.max() - att_np.min())
                att_np_norm_uint = (att_np_norm * 255).astype(np.uint8)

                offsets_el = offsets_el.cpu().detach().numpy()

                # ------

                superimposed = self._make_superimposed(att_np_norm_uint, x_low_np, color_map, use_masked_superimposed)
                superimposed_list.append(superimposed)

                vision = self._make_vision(att_np_norm, x_low_np)
                vision_list.append(vision)

                chosen_samples = self._chosen_samples(offsets_el, x_low_np, self.x_high.shape[-2:],
                                                      patches.shape[-2:], dakken=0.25)
                chosen_samples_list.append(chosen_samples)

            image_list = [
                ["attention_map",   att,                    False],
                ["superimposed",    superimposed_list,      True],
                ["vision",          vision_list,            True],
                ["chosen_samples",  chosen_samples_list,    True]
            ]
            self._save_image_lists(epoch, image_list)

        if self.use_wandb and epoch > 0:
            self.wandb_logger(epoch, losses, metrics)

    def _save_image_lists(self, epoch, image_sets):
        images = {}
        for name, image_list, convert in image_sets:
            if convert:
                image_list = torch.from_numpy((np.array(image_list) / 255.).astype(np.float32))
                image_list = image_list.permute(0, 3, 1, 2)

            i_grid = torchvision.utils.make_grid(image_list, nrow=self.nrow, normalize=True, scale_each=True,
                                                 pad_value=1)

            self.tensorboard_and_image(name, i_grid, epoch, dataformats='CHW')
            images[name] = [transforms.ToPILImage()(i_grid)]

        self.save_images(epoch, images)
        self.send_images(epoch, images)

    @staticmethod
    def _make_superimposed(att, x_low, color_map, use_masked_superimposed):

        heat_map = cv2.applyColorMap(255 - att, color_map)

        superimposed = cv2.addWeighted(heat_map, 0.5, x_low, 0.5, 0)

        if use_masked_superimposed:
            _, thres = cv2.threshold(att, int(255 * 0.1), 255, cv2.THRESH_BINARY)
            fin = cv2.addWeighted(heat_map, 0.4, x_low, 0.6, 0)
            fin_masked = cv2.bitwise_and(fin, fin, mask=thres)
            x_low_np_masked = cv2.bitwise_and(x_low, x_low, mask=255 - thres)
            superimposed_masked = x_low_np_masked + fin_masked

            return superimposed_masked
        else:
            return superimposed

    @staticmethod
    def _make_vision(att, x_low):
        return (x_low * att[:, :, np.newaxis]).astype(np.uint8)

    @staticmethod
    def _chosen_samples(offsets, x_low, x_high_shape, window_size, dakken=0.25):
        offsets_scaled = offsets.copy()
        scale = np.array([x_low.shape[0] / x_high_shape[0], x_low.shape[1] / x_high_shape[1]])
        offsets_scaled[:] = offsets_scaled[:] * scale
        window_size = (window_size * scale).astype(int)

        x_c = x_low.copy()
        x_c = x_c * dakken
        for of in offsets_scaled:
            x_from = max([0, of[0]])
            x_to = min([of[0] + window_size[0], x_c.shape[0]])
            y_from = max([0, of[1]])
            y_to = min([of[1] + window_size[1], x_c.shape[1]])

            x_c[x_from: x_to, y_from: y_to] = x_low[x_from: x_to, y_from: y_to]

        return x_c

    def _add_metric(self, metric, epoch, split):
        if self.write_tensorboard:
            for metric_name in metric:
                self.writer.add_scalar(f'{metric_name.capitalize()}/{split.capitalize()}', metric[metric_name], epoch)
