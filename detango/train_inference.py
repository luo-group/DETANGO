import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import os
import argparse
from tqdm import tqdm
import pickle
import yaml

from transformers import get_linear_schedule_with_warmup

import sys
sys.path.append('.')
from detango.data import DetangoDataset
from detango.model import DetangoModel
from detango.utils import DetangoLoss, evaluate_metrics_spearman

import warnings
warnings.filterwarnings("ignore")


torch.set_num_threads(1)


def train(model, device, dataloader, loss_fn, optimizer, epoch, writer):

    model.train()

    current_lr = optimizer.param_groups[0]['lr']
    print(f'Epoch {epoch+1}, LR: {current_lr}')

    total_loss = 0.0

    for step, data in tqdm(enumerate(dataloader), desc=f'Training {epoch}'):
        
        emb, pos, wts, esm1v, stability, mask = data
        emb = emb.to(device)
        pos = pos.to(device)
        wts = wts.to(device)
        esm1v = esm1v.to(device)
        stability = stability.to(device)
        mask = mask.to(device)

        stability_pred, function_plausibility, stability_plausibility = model(emb, pos, stability)
        esm1v_fit = function_plausibility + stability_plausibility

        loss, loss_dict = loss_fn(esm1v, esm1v_fit, stability, stability_pred, function_plausibility, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        for n,l in loss_dict.items():
            writer.add_scalar(f'Train/{n}', l, epoch*len(dataloader)+step)

    total_loss /= len(dataloader)

    return total_loss


def evaluate(model, device, dataloader, loss_fn, epoch, writer, istest=False):

    model.eval()

    total_loss = 0.0
    
    esm1v_list, esm1v_fit_list, minus_ddg_list, minus_ddg_pred_list = [], [], [], []
    q_list = []
    mask_list = []
    pos_list = []
    with torch.no_grad():
        for step, data in tqdm(enumerate(dataloader), desc=f'Evaluating {epoch}'):
        
            emb, pos, wts, esm1v, minus_ddg, mask = data
            emb = emb.to(device)
            pos = pos.to(device)
            wts = wts.to(device)
            esm1v = esm1v.to(device)
            minus_ddg = minus_ddg.to(device)
            mask = mask.to(device)

            s, q, ss = model(emb, pos, minus_ddg)
            minus_ddg_pred = s
            esm1v_fit = ss + q

            esm1v_list.append(esm1v)
            esm1v_fit_list.append(esm1v_fit)
            minus_ddg_list.append(minus_ddg)
            minus_ddg_pred_list.append(minus_ddg_pred)
            q_list.append(q)
            pos_list.append(pos)
            mask_list.append(mask)

        esm1v = torch.cat(esm1v_list, dim=0)
        esm1v_fit = torch.cat(esm1v_fit_list, dim=0)
        minus_ddg = torch.cat(minus_ddg_list, dim=0)
        minus_ddg_pred = torch.cat(minus_ddg_pred_list, dim=0)
        mask = torch.cat(mask_list, dim=0)

        q = torch.cat(q_list, dim=0)
        q_pred = torch.cat(q_list, dim=0)
        q_pred = 0-q_pred.mean(axis=1).cpu().detach().numpy()

        loss, loss_dict = loss_fn(esm1v, esm1v_fit, minus_ddg, minus_ddg_pred, q, mask)
        total_loss += loss.item()

        sr_s = evaluate_metrics_spearman(minus_ddg.cpu().detach().numpy(), minus_ddg_pred.cpu().detach().reshape(-1))

        if istest:
            for n,l in loss_dict.items():
                writer.add_scalar(f'Test/{n}', l, epoch)
            writer.add_scalar('Test/sr_s', sr_s, epoch)
        
        else:
            for n,l in loss_dict.items():
                writer.add_scalar(f'Val/{n}', l, epoch)
            writer.add_scalar('Val/sr_s', sr_s, epoch)

    total_loss /= len(dataloader)

    results = dict()

    return loss_dict['loss'], pos_list, q_list, results


def main():

    parser = argparse.ArgumentParser(description='Train, set hyperparameters')
    parser.add_argument('--config', type=str, default='config/training.yaml', help='config file')
    parser.add_argument('--protein', '-p', type=str, help='protein name')
    parser.add_argument('--sequence_wt', type=str, help='wild-type sequence')
    parser.add_argument('--stability_col', type=str, default='spurs', help='the column name for stability in the csv file')
    parser.add_argument('--sample_seed', type=int, default=0, help='the sample seed for dataset')
    parser.add_argument('--cuda_device', '-c', type=int, default=0, help='CUDA device')
    args = parser.parse_args()

    '''Load config'''
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    for k, v in config.items():
        setattr(args, k, v)

    '''Set random seed'''
    seed = args.sample_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    print(f'*** Fold {seed} ***')
    
    '''Prepare model'''
    device = torch.device(f'cuda:{args.cuda_device}' if torch.cuda.is_available() else 'cpu')
    model = DetangoModel(args)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup_steps = int(0.05*args.max_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=args.max_epochs
    )

    '''Prepare dataset'''
    print("Preparing dataset...")
    dataset = DetangoDataset(protein=args.protein, sequence_wt=args.sequence_wt, stability_col=args.stability_col)
    kfold = KFold(n_splits=10, shuffle=True, random_state=0)
    for i, (train_idx, test_idx) in enumerate(kfold.split(dataset)):
        if i == seed:
            print(f'Executing fold {i}')
            train_dataset = Subset(dataset, train_idx)
            val_dataset = Subset(dataset, test_idx)
    print(len(dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    writer = SummaryWriter(log_dir=f'log/{args.protein}_{args.sample_seed}')

    '''Create result dirs'''
    if not os.path.exists('results'):
        os.makedirs('results')
    if not os.path.exists(f'results/{args.protein}'):
        os.makedirs(f'results/{args.protein}')
    if not os.path.exists(f'results/{args.protein}/intermediates'):
        os.makedirs(f'results/{args.protein}/intermediates')

    '''Training'''
    loss_fn = DetangoLoss()
    best_mse = np.inf
    endure = 0
    best_epoch = 0

    for epoch in range(args.max_epochs):
    
        loss_train = train(model, device, train_loader, loss_fn, optimizer, epoch, writer)
        print(f'Train loss: {loss_train}')

        loss_val, pos_list, q_list, _ = evaluate(model, device, val_loader, loss_fn, epoch, writer)
        print(f'Val loss: {loss_val}')

        # loss_test, pos_list, q_list, _ = evaluate(model, device, test_loader, loss_fn, optimizer, epoch, writer, istest=True)
        # print(f'Test loss: {loss_test}')

        if loss_val < best_mse:
            best_mse = loss_val
            endure = 0
            torch.save(model.state_dict(), f'results/{args.protein}/intermediates/{args.sample_seed}_best_model.pth')
        else:
            endure += 1
            if endure >= args.endure:
                print('Early stop!')
                break

        scheduler.step()

    '''Inference'''
    del model
    model = DetangoModel(args)
    model.load_state_dict(torch.load(f'results/{args.protein}/intermediates/{args.sample_seed}_best_model.pth', weights_only=True))
    model = model.to(device)

    loss_test, pos_list, q_list, _ = evaluate(model, device, test_loader, loss_fn, epoch, writer, istest=True)

    print(f'Test loss: {loss_test}')
    print(args.protein)

    q_pred = torch.cat(q_list, dim=0).cpu().detach()
    poses = torch.cat(pos_list, dim=0).cpu().detach()
    sequence_wt = dataset.sequence_wt

    mutant_list, q_list = [], []
    for pos,wt in zip(poses, sequence_wt):
        for i,aa in enumerate(dataset.alphabet):
            if aa != wt:
                mutant_list.append(f'{wt}{pos.item()}{aa}')
                q_list.append(q_pred[pos,i].item())
    df = pd.DataFrame({'mutant': mutant_list, 'q': q_list})

    df.to_csv(f'results/{args.protein}/intermediates/detango_{args.stability_col}_{args.sample_seed}.csv', index=False)


if __name__ == '__main__':
    main()