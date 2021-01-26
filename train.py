import os
from glob import glob

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from configurate import get_config
from dataloader import get_dataloader
from protonets import ProtoNet
from prototypical_loss import PrototypicalLoss
from one_cycle_policy import OneCyclePolicy
from utils import AverageMeter

best_acc1 = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def main():
    global args, best_acc1, device
    args = get_config()

    # Init seed
    torch.cuda.cudnn_enabled = False
    np.random.seed(args.manual_seed)
    torch.manual_seed(args.manual_seed)
    torch.cuda.manual_seed(args.manual_seed)

    train_loader, val_loader = get_dataloader(args, 'train', 'val')

    model = ProtoNet().to(device)

    criterion = PrototypicalLoss().to(device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    writer = SummaryWriter(args.log_dir)
    cudnn.benchmark = True

    if args.resume:
        checkpoint = torch.load(sorted(glob(f'{args.log_dir}/checkpoint_*.pth'), key=len)[-1])
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        best_acc1 = checkpoint['best_acc1']

        # scheduler = OneCyclePolicy(optimizer, args.lr, (args.epochs - start_epoch)*args.iterations)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    gamma=args.lr_scheduler_gamma,
                                                    step_size=args.lr_scheduler_step)
        print(f"load checkpoint {args.exp_name}")
    else:
        start_epoch = 0
        # scheduler = OneCyclePolicy(optimizer, args.lr, args.epochs*args.iterations)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer,
                                                    gamma=args.lr_scheduler_gamma,
                                                    step_size=args.lr_scheduler_step)

    print(f"model parameter : {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    for epoch in range(start_epoch, args.epochs):

        train_loss = train(train_loader, model, optimizer, criterion)
        val_loss, acc1 = validate(val_loader, model, criterion)

        if acc1 >= best_acc1:
            is_best = True
            best_acc1 = acc1
        else:
            is_best = False

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer_state_dict': optimizer.state_dict(),
        }, is_best)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Acc/Top1", acc1, epoch)

        print(f"[{epoch}/{args.epochs}] {train_loss:.3f}, {val_loss:.3f}, {acc1:.3f}, # {best_acc1:.3f}")

        scheduler.step()

    writer.close()


def train(train_loader, model, optimizer, criterion):
    losses = AverageMeter()
    num_support = args.num_support_tr

    # switch to train mode
    model.train()
    for i, data in enumerate(train_loader):
        input, target = data[0].to(device), data[1].to(device)

        output = model(input)
        loss, _ = criterion(output, target, num_support)

        losses.update(loss.item(), input.size(0))

        # compute gradient and do optimize step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return losses.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    num_support = args.num_support_val

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            input, target = data[0].to(device), data[1].to(device)

            output = model(input)
            loss, acc1 = criterion(output, target, num_support)

            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))

    return losses.avg, top1.avg


def save_checkpoint(state, is_best):
    directory = args.log_dir
    filename = directory + f"/checkpoint_{state['epoch']}.pth"

    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(state, filename)

    if is_best:
        filename = directory + "/model_best.pth"
        torch.save(state, filename)


if __name__ == '__main__':
    main()
