import torch 
import torch.nn as nn
from torch.utils.data import Dataset

import argparse
import numpy as np

from train_lbc import encoder_trainer, predictor_trainer
import wandb


class circuit_data(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return (self.data.shape[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label

def main(args):
    ## Write logs using wandb
    # wandb initialization
    wandb.init(project="LbC", name="PRE-LbC-ep{}-b{}-em{}-temp{}-head".format(args.enc_epochs, args.batch_size, args.embed_size, args.temperature))

    device = 'cuda:7'
    training_size = 500
    # loading data
    data = np.load(args.data_path)
    labels = np.load(args.label_path)

    data = torch.as_tensor(data, dtype=torch.float)
    labels = torch.as_tensor(labels, dtype=torch.float)
    labels = nn.functional.normalize(labels, dim=0)
    print(torch.mean(labels,dim=0))

    train_data = data[:training_size]
    train_label = labels[:training_size]#.view(training_size, -1)
    test_data = data[training_size:]
    test_label = labels[training_size:]#.view(training_size, -1)

    dataset_train = circuit_data(train_data, train_label)
    dataset_test = circuit_data(test_data, test_label)

    input_size = data.size(-1)
    target_size = train_label.size(-1)

    # encoder pretraining
    enc_trainer = encoder_trainer(num_epoch=args.enc_epochs, batch_size=args.batch_size, 
                                  temp=args.temperature, thre=args.threshold,
                                  learning_rate=args.lr_enc, 
                                  save_freq=args.print_freq, device=device,
                                  embed_size=args.embed_size, input_size=input_size)
    best_model_path = enc_trainer.train(dataset_train)

    # predictor training
    pred_trainer = predictor_trainer(num_epoch=args.pred_epochs, batch_size=args.batch_size, 
                                  learning_rate=args.lr_pred, save_freq=10, 
                                  pretrained=best_model_path, device=device,
                                  embed_size=args.embed_size, input_size=input_size, 
                                  target_size=target_size, tuning_full=args.tune_full)
    pred_trainer.train(dataset_train,dataset_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--lr_enc', default=5e-4, type=float, help='initial encoder learning rate')
    parser.add_argument('--lr_pred', default=5e-4, type=float, help='initial predictor learning rate')
    parser.add_argument('--batch-size', default=128, type=int, help='mini-batch size')
    parser.add_argument('--temperature', default=2.0, type=float, help='temperature for cosine similarity')
    parser.add_argument('--threshold', default=0.1, type=float, help='smallest label distance to form postive pairs')
    parser.add_argument('--embed_size', default=40, type=int, help='number of embedding features')
    parser.add_argument('--enc_epochs', default=2000, type=int, help='number of epochs for encoder training')
    parser.add_argument('--pred_epochs', default=3000, type=int, help='number of epochs for predictor training')
    parser.add_argument('--print_freq', default=100, type=int, help='frequency of printing training log & saving model')
    parser.add_argument('--data_path', type=str, default="inputdata.npy", help="path to data")
    parser.add_argument('--label_path', type=str, default="outputdata.npy", help="path to labels")
    parser.add_argument('--tune_full', action='store_false', help="Tuning the full model instead of the head only, default: head only")
    args = parser.parse_args()
    print(args.tune_full)
    main(args)