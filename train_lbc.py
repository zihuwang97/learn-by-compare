import os
import time
import datetime
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import wandb

from models import encoder, predictor, whole_model
from util import mkdir


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


def LbC_loss(rep, labels, temp=2.0, thre=0.1):
    bsz = rep.size(0)
    # calculating label distance and sort
    # labels = labels.view(1,-1)
    # label_dist = torch.abs(labels - labels.T)
    label_dist = torch.cdist(labels, labels, p=1)
    label_dist_sorted, indices = torch.sort(label_dist,-1,descending=True)

    # calculating logits and sorting them w.r.t. label distance
    scores = rep @ rep.T
    logits = torch.exp(scores / temp)
    logits_sorted = torch.scatter(torch.zeros_like(logits).cuda(rep.device), 
                                  -1, torch.argsort(indices), logits)

    # generate mask out long distance positives
    mask = torch.where(label_dist_sorted <= thre, 1.0, 0.0).cuda(rep.device)
    elig_pos_count = torch.clamp(torch.sum(mask, -1), min=1.0)

    # loss
    neg_sum = torch.cumsum(logits_sorted, -1)
    loss = torch.log(logits_sorted[:,1:] / neg_sum[:,:-1])
    loss = torch.where(mask[:,1:] == 1, loss, 0.0) # mask out long distance positives
    loss = torch.sum(loss, -1) / elig_pos_count
    # loss = torch.sum(loss, -1) / (bsz - 1)
    return - torch.sum(loss) / bsz


class encoder_trainer():
    def __init__(self, num_epoch, batch_size, temp, thre, learning_rate, save_freq, device, embed_size, input_size):
        self.num_epochs = num_epoch
        self.no_btchs = batch_size
        self.temperature = temp
        self.thre = thre
        self.lr = learning_rate
        self.save_freq = save_freq
        self.device = device
        self.embed_size = embed_size
        self.input_size = input_size
    def train(self, dataset_train):
        # train = torch.as_tensor(train, dtype=torch.float)
        # dataset_train = circuit_data(train)
        model = encoder(input_feature_num=self.input_size, hdn_size=self.embed_size).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        trainloader = DataLoader(dataset_train, batch_size=self.no_btchs,
                                    shuffle=True, num_workers=0, pin_memory=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader))

        ### training
        ## Phase 1: representation learning
        # making saving directory
        save_path = os.path.join(os.getcwd(), 'train_log')
        today = datetime.date.today()
        formatted_today = today.strftime('%y%m%d')
        now = time.strftime("%H:%M:%S")
        save_path = os.path.join(save_path, formatted_today + now)
        mkdir(save_path)
        training_loss = []
        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0
            for i, (sample,labels) in enumerate(trainloader, 0):
                model.zero_grad()
                sample = sample.to(self.device)
                labels = labels.to(self.device)
                rep = model(sample) 
                rep = nn.functional.normalize(rep,dim=-1)
                loss = LbC_loss(rep, labels, temp=self.temperature, thre=self.thre)
                loss.backward()
                optimizer.step()
            scheduler.step()
            if epoch % 10 == 0:
                wandb.log({"Training loss": loss})

            if (epoch + 1) % self.save_freq == 0:
                print('[%d]  loss: %.3f' % (epoch + 1, loss))
                training_loss.append(loss.detach())
                model_state = model.state_dict()
                file_save_path = os.path.join(save_path, 'checkpoint_{:04d}.pth.tar'.format(epoch))
                torch.save(model_state, file_save_path)
        return file_save_path


class predictor_trainer():
    def __init__(self, num_epoch, batch_size, learning_rate, save_freq, pretrained, device, embed_size, input_size, target_size, tuning_full):
        self.num_epochs = num_epoch
        self.no_btchs = batch_size
        self.lr = learning_rate
        self.save_freq = save_freq
        self.pretrained = pretrained  # pretrained model path
        self.device = device
        self.embed_size = embed_size
        self.input_size = input_size
        self.target_size = target_size
        self.tuning_full = tuning_full
    def train(self, dataset_train, dataset_test):
        # load the pre-trained encoder
        enc = encoder(input_feature_num=self.input_size, hdn_size=self.embed_size)
        state_dict = torch.load(self.pretrained, map_location="cpu")
        enc.load_state_dict(state_dict)
        enc.to(self.device)

        model = predictor(embed_size=self.embed_size, target_size=self.target_size).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        trainloader = DataLoader(dataset_train, batch_size=self.no_btchs,
                                    shuffle=True, num_workers=0, pin_memory=True)
        testloader = DataLoader(dataset_test, batch_size=500,
                                    shuffle=True, num_workers=0, pin_memory=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader))

        if self.tuning_full:
            optimizer_enc = torch.optim.Adam(enc.parameters(), lr=self.lr)
            scheduler_enc = lr_scheduler.CosineAnnealingLR(optimizer_enc, T_max=len(trainloader))
        else:
            optimizer_enc = None
            scheduler_enc = None

        criterion = nn.L1Loss()
        ### training
        ## Phase 2: training regressor
        # making saving directory
        save_path = os.path.join(os.getcwd(), 'train_log')
        today = datetime.date.today()
        formatted_today = today.strftime('%y%m%d')
        now = time.strftime("%H:%M:%S")
        save_path = os.path.join(save_path, formatted_today + now)
        mkdir(save_path)

        errors = []
        for epoch in range(self.num_epochs):
            model.train()
            enc.train()
            running_loss = 0
            for i, (sample,labels) in enumerate(trainloader, 0):
                model.zero_grad()
                if self.tuning_full: enc.zero_grad()
                sample = sample.to(self.device)
                labels = labels.to(self.device)
                with torch.no_grad():
                    rep = enc(sample) 
                pred = model(rep)
                loss = criterion(labels.view(-1,1),pred.view(-1,1))
                loss.backward()
                optimizer.step()
                if optimizer_enc: optimizer_enc.step()
            scheduler.step()
            if scheduler_enc: scheduler_enc.step()

            if (epoch + 1) % self.save_freq == 0:
                # print('[%d, %5d]  loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))
                model_state = model.state_dict()
                enc_state = enc.state_dict()
                file_save_path = os.path.join(save_path, 'pred_checkpoint_{:04d}.pth.tar'.format(epoch))
                torch.save(model_state, 'train_log/lbc_checkpoints_dly1/pred_checkpoint_{:04d}.pth.tar'.format(epoch))
                torch.save(enc_state, 'train_log/lbc_checkpoints_dly1/enc_checkpoint_{:04d}.pth.tar'.format(epoch))

            # testing model
            if (epoch + 1) % 10 == 0:
                model.eval()
                for step, (sample,labels) in enumerate(testloader, 0):
                    sample = sample.to(self.device)
                    labels = labels.to(self.device)
                    rep = enc(sample) 
                    pred = model(rep)
                    error = criterion(labels.view(-1,1), pred.view(-1,1))
                    # error1 = criterion(labels[:,0], pred[:,0])
                    # error2 = criterion(labels[:,1], pred[:,1])
                    if step % 50 == 0:
                        rand_num = random.randint(0,len(labels)-10)
                        # print(labels[rand_num:rand_num+5], pred.view(-1)[rand_num:rand_num+5], error)
                        print(error.item())#, error1.item(), error2.item())
                        errors.append(error)
                        wandb.log({"Test MAE error": error})
        # errors = torch.tensor(errors)
        # torch.save(errors, 'train_log/lbc_testingerror.pt')


class whole_trainer():
    def __init__(self, num_epoch, batch_size, learning_rate, save_freq, device, embed_size, input_size, target_size):
        self.num_epochs = num_epoch
        self.no_btchs = batch_size
        self.lr = learning_rate
        self.save_freq = save_freq
        self.device = device
        self.embed_size = embed_size
        self.input_size = input_size
        self.target_size = target_size
    def train(self, dataset_train, dataset_test):
        # load the pre-trained encoder
        model = whole_model(input_size=self.input_size, embed_size=self.embed_size, target_size=self.target_size).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        trainloader = DataLoader(dataset_train, batch_size=self.no_btchs,
                                    shuffle=True, num_workers=0, pin_memory=True)
        testloader = DataLoader(dataset_test, batch_size=500,
                                    shuffle=True, num_workers=0, pin_memory=True)
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader))
        criterion = nn.L1Loss()

        errors = []
        ### training
        for epoch in range(self.num_epochs):
            model.train()
            running_loss = 0
            for i, (sample,labels) in enumerate(trainloader, 0):
                model.zero_grad()
                sample = sample.to(self.device)
                labels = labels.to(self.device)
                pred = model(sample)
                loss = criterion(labels.view(-1,1),pred.view(-1,1))
                loss.backward()
                optimizer.step()
            scheduler.step()

            if (epoch + 1) % self.save_freq == 0:
                model_state = model.state_dict()
                file_save_path = os.path.join(os.getcwd(),'train_log/l1_checkpoints', 'checkpoint_{:04d}.pth.tar'.format(epoch))
                torch.save(model_state, file_save_path)
            #     print('[%d, %5d]  loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))
                
            # testing model
            if (epoch + 1) % 10 == 0:
                model.eval()
                for step, (sample,labels) in enumerate(testloader, 0):
                    sample = sample.to(self.device)
                    labels = labels.to(self.device)
                    pred = model(sample)
                    error = criterion(labels.view(-1,1), pred.view(-1,1))
                    # error1 = criterion(labels[:,0], pred[:,0])
                    # error2 = criterion(labels[:,1], pred[:,1])
                    neg_count = torch.where(labels<=0, 1.0, 0.0)
                    neg_count = torch.sum(neg_count)
                    if step % 50 == 0:
                        rand_num = random.randint(0,len(labels)-10)
                        # print(labels[rand_num:rand_num+5], pred.view(-1)[rand_num:rand_num+5], error, neg_count)
                        print(error.item())#, error1.item(), error2.item())
                        errors.append(error)
                        wandb.log({"Test MAE error": error.item()})
                        # wandb.log({"Test MAE error1": error1.item()})
                        # wandb.log({"Test MAE error2": error2.item()})

        # errors = torch.tensor(errors)
        # torch.save(errors, 'train_log/l1_testingerror.pt')



