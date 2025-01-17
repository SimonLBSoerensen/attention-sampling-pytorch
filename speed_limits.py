import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import time

from models.attention_model import AttentionModelTrafficSigns
from models.feature_model import FeatureModelTrafficSigns
from models.classifier import ClassificationHead

from ats.core.ats_layer import ATSModel
from ats.utils.regularizers import MultinomialEntropy
from ats.utils.logging import TrainingLogger, WandBLogger
from ats.utils.model_checkpoint import ModelCheckpoint, load_checkpoint

from dataset.speed_limits_dataset import SpeedLimits
from train import train, evaluate

def main(opts):
    train_dataset = SpeedLimits('dataset/traffic_data', train=True)
    train_loader = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True, num_workers=opts.num_workers)

    test_dataset = SpeedLimits('dataset/traffic_data', train=False)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=opts.batch_size, num_workers=opts.num_workers)

    attention_model = AttentionModelTrafficSigns(squeeze_channels=True, softmax_smoothing=1e-4)
    feature_model = FeatureModelTrafficSigns(in_channels=3, strides=[1, 2, 2, 2], filters=[32, 32, 32, 32])
    classification_head = ClassificationHead(in_channels=32, num_classes=len(train_dataset.CLASSES))

    ats_model = ATSModel(attention_model, feature_model, classification_head, n_patches=opts.n_patches,
                         patch_size=opts.patch_size)
    ats_model = ats_model.to(opts.device)
    optimizer = optim.Adam([{'params': ats_model.attention_model.part1.parameters(), 'weight_decay': 1e-5},
                            {'params': ats_model.attention_model.part2.parameters()},
                            {'params': ats_model.feature_model.parameters()},
                            {'params': ats_model.classifier.parameters()},
                            {'params': ats_model.sampler.parameters()},
                            {'params': ats_model.expectation.parameters()}
                            ], lr=opts.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.decrease_lr_at, gamma=0.1)

    run_folder = os.path.join(opts.output_dir, opts.run_name)
    logger = TrainingLogger(run_folder, ats_model, test_dataset, use_wandb=opts.use_wandb, project="traffic_data",
                            make_images_every=opts.make_images_every, send_images_every=opts.send_images_every)

    model_folder = os.path.join(run_folder, "saves")
    model_checkpoint = ModelCheckpoint(model_folder, ats_model, optimizer,
                                       save_best=opts.save_best, save_frequency=opts.save_frequency)


    class_weights = train_dataset.class_frequencies
    class_weights = torch.from_numpy((1. / len(class_weights)) / class_weights).to(opts.device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    entropy_loss_func = MultinomialEntropy(opts.regularizer_strength)

    for epoch in range(opts.epochs):
        train_loss, train_metrics = train(ats_model, optimizer, train_loader,
                                          criterion, entropy_loss_func, opts)

        with torch.no_grad():
            test_loss, test_metrics = evaluate(ats_model, test_loader, criterion,
                                               entropy_loss_func, opts)

        logger(epoch, (train_loss, test_loss), (train_metrics, test_metrics), return_images=True)


        model_checkpoint(epoch, test_loss)

        # Perform scheduler step
        scheduler.step()

#
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--regularizer_strength", type=float, default=0.05,
                        help="How strong should the regularization be for the attention")
    parser.add_argument("--softmax_smoothing", type=float, default=1e-4,
                        help="Smoothing for calculating the attention map")
    parser.add_argument("--lr", type=float, default=0.001, help="Set the optimizer's learning rate")
    parser.add_argument("--n_patches", type=int, default=5, help="How many patches to sample")
    parser.add_argument("--patch_size", type=int, default=100, help="Patch size of a square patch")
    parser.add_argument("--batch_size", type=int, default=32, help="Choose the batch size for SGD")
    parser.add_argument("--epochs", type=int, default=5000, help="How many epochs to train for")
    parser.add_argument("--decrease_lr_at", type=float, default=-2, help="Decrease the learning rate in this epoch, "
                                                                         "If -2 the decrease will happen at epochs//2 "
                                                                         "and -1 will turn it off")
    parser.add_argument("--clipnorm", type=float, default=1, help="Clip the norm of the gradients")
    parser.add_argument("--output_dir", type=str, help="An output directory", default='output/traffic')
    parser.add_argument('--run_name', type=str, default='run')
    parser.add_argument('--save_best', type=bool, default=True)
    parser.add_argument('--use_wandb', type=bool, default=True)
    parser.add_argument("--save_frequency", type=int, default=500, help="How many epochs between each save")
    parser.add_argument("--make_images_every", type=int, default=10, help="How many epochs between each image log")
    parser.add_argument("--send_images_every", type=int, default=100, help="How many epochs between each image log is send to wandb")
    parser.add_argument('--num_workers', type=int, default=30, help='Number of workers to use for data loading')

    opts = parser.parse_args()
    opts.run_name = f"{opts.run_name}_{time.strftime('%Y%m%dT%H%M%S')}"
    opts.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if opts.decrease_lr_at == -1:
        opts.decrease_lr_at = opts.epochs + 10
    elif opts.decrease_lr_at == -2:
        opts.decrease_lr_at = opts.epochs//2

    main(opts)
