import argparse
import os
from sys import platform

from torch.utils.data import DataLoader
from torch.autograd import Variable

from models import *
from datasets import *

import torch.nn as nn
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from models.mvssnet import get_mvss


def dice_loss(gt, out, smooth=1.0):
    gt = gt.view(-1)
    out = out.view(-1)

    intersection = (gt * out).sum()
    dice = (2.0 * intersection + smooth) / (torch.square(gt).sum() + torch.square(
        out).sum() + smooth)  # TODO: need to confirm this matches what the paper says, and also the calculation/result is correct

    return 1.0 - dice


def bgr_to_rgb(t):
    b, g, r = torch.unbind(t, 1)
    return torch.stack((r, g, b), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--train_paths_file", type=str, default="../FaceForensics/train_files.txt",
                        help="path to the file with training input paths")  # each line of this file should contain "/path/to/image.ext /path/to/mask.ext /path/to/edge.ext 1 (for fake)/0 (for real)"; for real image.ext, set /path/to/mask.ext and /path/to/edge.ext as a string None
    parser.add_argument("--valid_paths_file", type=str, default="../FaceForensics/valid_files.txt", 
                        help="path to the file with validation input paths")
    parser.add_argument("--image_size", type=int, default=512, help="size of the images")
    parser.add_argument("--batch_size", type=int, default=12,
                        help="size of the batches")  # no default value given by paper
    parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
    parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--workers", type=int, default=4, help="number of cpu threads to use during batch generation")
    parser.add_argument("--channels", type=int, default=3, help="number of image channels")
    parser.add_argument('--decay_epoch', type=int, default=50, help='decay')
    parser.add_argument("--lambda_seg", type=float, default=0.16, help="pixel-scale loss weight (alpha)")
    parser.add_argument("--lambda_clf", type=float, default=0.04, help="image-scale loss weight (beta)")
    parser.add_argument("--run_name", type=str, default="MVSS-Net", help="run name")
    parser.add_argument("--log_interval", type=int, default=100, help="interval between saving image samples")
    parser.add_argument("--checkpoint_interval", type=int, default=1000,
                        help="batch interval between model checkpoints")
    parser.add_argument('--load_path', type=str, default=None, help='pretrained model or checkpoint for continued training')
    parser.add_argument('--nGPU', type=int, default=1, help='number of gpus')  # TODO: multiple GPU support
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = get_mvss(backbone='resnet50',
                     pretrained_base=True,
                     nclass=1,
                     sobel=True,
                     constrain=True,
                     n_input=args.channels,
                     ).to(device)

    # Losses that are built-in in PyTorch
    criterion_clf = nn.BCEWithLogitsLoss().to(device)

    # Load pretrained models
    if args.load_path != None:
        print('Load pretrained model: ' + args.load_path)
        model.load_state_dict(torch.load(args.load_path))

    # Tensor
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

    # Time for log
    logtm = datetime.now().strftime("%Y%m%d%H%M%S")

    # Dataset
    train_dataset = Datasets(args.train_paths_file, args.image_size)
    valid_dataset = Datasets(args.valid_paths_file, args.image_size)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, 
                                  pin_memory=True, drop_last=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, num_workers=args.workers, shuffle=False, 
                                  pin_memory=True, drop_last=True)

    # Conversion from epoch to step/iter
    decay_iter = args.decay_epoch * len(train_dataloader)

    # Optimizers
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(args.b1, args.b2))
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                   step_size=decay_iter,
                                                   gamma=0.5)

    # ----------
    #  Training
    # ----------
    os.makedirs("logs", exist_ok=True)
    writer = SummaryWriter("logs/" + logtm + "_" + args.run_name)
    checkpoint_dir = "checkpoints/" + logtm + "_" + args.run_name
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(args.epoch, args.n_epochs):
        print("Starting Epoch ", epoch + 1)

        # Record loss sum for training and validation per epoch by summing over dataloader
        # training
        train_epoch_total_seg = 0    # Pixel-scale loss
        train_epoch_total_clf = 0    # Image-scale loss
        train_epoch_total_edg = 0    # Edge loss
        train_epoch_total_model = 0  # Total loss
        train_epoch_steps = 0        # Track total number of iterations

        # validation
        valid_epoch_total_seg = 0    # Pixel-scale loss
        valid_epoch_total_clf = 0    # Image-scale loss
        valid_epoch_total_edg = 0    # Edge loss
        valid_epoch_total_model = 0  # Total loss
        valid_epoch_steps = 0        # Track total number of iterations


        # Iterate over train_loader
        for step, data in enumerate(train_dataloader):
            curr_steps = epoch * len(train_dataloader) + step + 1

            # Read from train_dataloader
            train_in_imgs = Variable(data["input"].type(Tensor))
            train_in_masks = Variable(data["mask"].type(Tensor))
            train_in_edges = Variable(data["edge"].type(Tensor))
            train_in_labels = Variable(data["label"].type(Tensor))

            # ------------------
            #  Train Generators
            # ------------------

            optimizer.zero_grad()

            # Prediction
            train_out_edges, train_out_masks = model(train_in_imgs)
            train_out_edges = torch.sigmoid(train_out_edges)
            train_out_masks = torch.sigmoid(train_out_masks)

            # Pixel-scale loss
            loss_seg = dice_loss(train_in_masks, train_out_masks)

            # Edge loss
            # TODO: is it the same as the paper?
            loss_edg = dice_loss(train_in_edges, train_out_edges)

            # Image-scale loss (with GMP)
            # TODO: GeM from MVSS-Net++
            gmp = nn.MaxPool2d(args.image_size)
            out_labels = gmp(train_out_masks).squeeze()
            loss_clf = criterion_clf(train_in_labels, out_labels)

            # Total loss
            alpha = args.lambda_seg
            beta = args.lambda_clf

            weighted_loss_seg = alpha * loss_seg
            weighted_loss_clf = beta * loss_clf
            weighted_loss_edg = (1.0 - alpha - beta) * loss_edg

            loss = weighted_loss_seg + weighted_loss_clf + weighted_loss_edg

            # backward prop and step
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # log losses for epoch
            train_epoch_steps += 1

            train_epoch_total_seg += weighted_loss_seg
            train_epoch_total_clf += weighted_loss_clf
            train_epoch_total_edg += weighted_loss_edg
            train_epoch_total_model += loss

            # --------------
            #  Log Progress (for certain steps)
            # --------------
            if step % args.log_interval == 0:
                print(f"[Epoch {epoch + 1}/{args.n_epochs}] [Batch {step + 1}/{len(train_dataloader)}] "
                      f"[Total Loss {loss:.3f}]"
                      f"[Pixel-scale Loss {weighted_loss_seg:.3e}]"
                      f"[Edge Loss {weighted_loss_edg:.3e}]"
                      f"[Image-scale Loss {weighted_loss_clf:.3e}]"
                      f"")

                writer.add_scalar("LearningRate", lr_scheduler.get_last_lr()[0],
                                  curr_steps)
                writer.add_scalar("Loss/Total Loss", loss, epoch * len(train_dataloader) + step)
                writer.add_scalar("Loss/Pixel-scale", weighted_loss_seg, curr_steps)
                writer.add_scalar("Loss/Edge", weighted_loss_edg, curr_steps)
                writer.add_scalar("Loss/Image-scale", weighted_loss_clf, curr_steps)

                in_imgs_rgb = bgr_to_rgb(train_in_imgs.clone().detach())
                writer.add_images('Input Img', in_imgs_rgb, epoch * len(train_dataloader) + step)

                writer.add_images('Input Mask', train_in_masks, epoch * len(train_dataloader) + step)
                writer.add_images('Output Mask', train_out_masks, epoch * len(train_dataloader) + step)
                writer.add_images('Input Edge', train_in_edges, epoch * len(train_dataloader) + step)
                writer.add_images('Output Edge', train_out_edges, epoch * len(train_dataloader) + step)

            # save model parameters
            # TODO: you can change when the parameters are saved
            if step % args.checkpoint_interval == 0:
                tm = datetime.now().strftime("%Y%m%d%H%M%S")
                torch.save(model.state_dict(),
                           os.path.join(checkpoint_dir, tm + '_' + args.run_name + '_' + str(
                               epoch + 1) + "_" + str(step + 1) + '.pth'))

        # Iterate over valid_loader
        with torch.no_grad():

            for step, data in enumerate(valid_dataloader):
                
                # Read from train_dataloader
                valid_in_imgs = Variable(data["input"].type(Tensor))
                valid_in_masks = Variable(data["mask"].type(Tensor))
                valid_in_edges = Variable(data["edge"].type(Tensor))
                valid_in_labels = Variable(data["label"].type(Tensor))

                # Prediction
                valid_out_edges, valid_out_masks = model(valid_in_imgs)
                valid_out_edges = torch.sigmoid(valid_out_edges)
                valid_out_masks = torch.sigmoid(valid_out_masks)

                # Pixel-scale loss
                loss_seg = dice_loss(valid_in_masks, valid_out_masks)

                # Edge loss
                loss_edg = dice_loss(valid_in_edges, valid_out_edges)

                # Image-scale loss (with GMP)
                gmp = nn.MaxPool2d(args.image_size)
                out_labels = gmp(valid_out_masks).squeeze()
                loss_clf = criterion_clf(valid_in_labels, out_labels)

                # Total loss
                alpha = args.lambda_seg
                beta = args.lambda_clf

                weighted_loss_seg = alpha * loss_seg
                weighted_loss_clf = beta * loss_clf
                weighted_loss_edg = (1.0 - alpha - beta) * loss_edg

                loss = weighted_loss_seg + weighted_loss_clf + weighted_loss_edg

                # log losses for epoch
                valid_epoch_steps += 1

                valid_epoch_total_seg += weighted_loss_seg
                valid_epoch_total_clf += weighted_loss_clf
                valid_epoch_total_edg += weighted_loss_edg
                valid_epoch_total_model += loss


        # --------------
        #  Log Progress (for epoch)
        # --------------

        # Training loss average for epoch
        if (train_epoch_steps != 0):
            train_epoch_avg_seg = train_epoch_total_seg / train_epoch_steps
            train_epoch_avg_edg = train_epoch_total_edg / train_epoch_steps
            train_epoch_avg_clf = train_epoch_total_clf / train_epoch_steps
            train_epoch_avg_model = train_epoch_total_model / train_epoch_steps
            
            print(f"[Epoch {epoch + 1}/{args.n_epochs}]"
                  f"[===== Train set ====]"
                  f"[Epoch Total Loss {train_epoch_avg_model:.3f}]"
                  f"[Epoch Pixel-scale Loss {train_epoch_avg_seg:.3e}]"
                  f"[Epoch Edge Loss {train_epoch_avg_edg:.3e}]"
                  f"[Epoch Image-scale Loss {train_epoch_avg_clf:.3e}]"
                  f"")

            writer.add_scalar("Epoch LearningRate", lr_scheduler.get_last_lr()[0],
                              epoch)

            in_imgs_rgb = bgr_to_rgb(train_in_imgs.clone().detach())
            writer.add_images('Epoch Input Img (train set)', in_imgs_rgb, epoch)

            writer.add_images('Epoch Input Mask (train set)', train_in_masks, epoch)
            writer.add_images('Epoch Output Mask (train set)', train_out_masks, epoch)
            writer.add_images('Epoch Input Edge (train set)', train_in_edges, epoch)
            writer.add_images('Epoch Output Edge (train set)', train_out_edges, epoch)


        # Validation loss average for epoch
        if (valid_epoch_steps != 0):
            valid_epoch_avg_seg = valid_epoch_total_seg / valid_epoch_steps
            valid_epoch_avg_edg = valid_epoch_total_edg / valid_epoch_steps
            valid_epoch_avg_clf = valid_epoch_total_clf / valid_epoch_steps
            valid_epoch_avg_model = valid_epoch_total_model / valid_epoch_steps

            print(f"[Epoch {epoch + 1}/{args.n_epochs}]"
                  f"[===== Validation set ====]"
                  f"[Epoch Total Loss {valid_epoch_avg_model:.3f}]"
                  f"[Epoch Pixel-scale Loss {valid_epoch_avg_seg:.3e}]"
                  f"[Epoch Edge Loss {valid_epoch_avg_edg:.3e}]"
                  f"[Epoch Image-scale Loss {valid_epoch_avg_clf:.3e}]"
                  f"")

            in_imgs_rgb = bgr_to_rgb(valid_in_imgs.clone().detach())
            writer.add_images('Epoch Input Img (valid set)', in_imgs_rgb, epoch)

            writer.add_images('Epoch Input Mask (valid set)', valid_in_masks, epoch)
            writer.add_images('Epoch Output Mask (valid set)', valid_out_masks, epoch)
            writer.add_images('Epoch Input Edge (valid set)', valid_in_edges, epoch)
            writer.add_images('Epoch Output Edge (valid set)', valid_out_edges, epoch)

        # Write train and validation loss
        writer.add_scalars('Epoch Loss/Total Loss', 
                    {'train': train_epoch_avg_model, 
                    'valid': valid_epoch_avg_model}, epoch)
        writer.add_scalars('Epoch Loss/Pixel-scale', 
                    {'train': train_epoch_avg_seg, 
                    'valid': valid_epoch_avg_seg}, epoch)
        writer.add_scalars('Epoch Loss/Edge', 
                    {'train': train_epoch_avg_edg, 
                    'valid': valid_epoch_avg_edg}, epoch)
        writer.add_scalars('Epoch Loss/Image-scale', 
                    {'train': train_epoch_avg_clf, 
                    'valid': valid_epoch_avg_clf}, epoch)

    print("Finished training")


if platform == "win32":
    if __name__ == '__main__':
        main()
else:
    main()
