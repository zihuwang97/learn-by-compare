import torch 
import torch.nn as nn
from torch.utils.data import Dataset

import argparse
import numpy as np

from train_lbc import encoder_trainer, predictor_trainer, whole_trainer
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
    wandb.init(project="LbC", name="PRE-L1-ep{}-b{}-em{}-fullmodel".format(args.epochs, args.batch_size, args.embed_size))
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

    # training
    trainer = whole_trainer(num_epoch=args.epochs, batch_size=args.batch_size, 
                            learning_rate=args.lr, save_freq=10, 
                            device=device, embed_size=args.embed_size,
                            input_size=input_size, target_size=target_size,)
    
    trainer.train(dataset_train,dataset_test)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch LbC Training')
    parser.add_argument('--lr', default=5e-4, type=float, help='initial encoder learning rate')
    parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size')
    parser.add_argument('--embed_size', default=40, type=int, help='number of embedding features')
    parser.add_argument('--epochs', default=3000, type=int, help='number of epochs for predictor training')
    parser.add_argument('--print_freq', default=100, type=int, help='frequency of printing training log & saving model')
    parser.add_argument('--data_path', type=str, default="inputdata.npy", help="path to data")
    parser.add_argument('--label_path', type=str, default="outputdata.npy", help="path to labels")

    args = parser.parse_args()

    main(args)




