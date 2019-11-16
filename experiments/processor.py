# !/usr/bin/env python
# -*- coding:utf-8 -*-

import os
import yaml
import time
import torch   
import torch.nn as nn
import pickle
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd

from codes.utils.utils import import_class,print_toolbar, end_toolbar
from configs.configs import GlobalConfig
from collections import OrderedDict
from torch.autograd import Variable
from codes.nets.losses import LipSeqLoss

gc = GlobalConfig()

class Processor():
    def __init__(self, arg):
        assert arg.phase in ('train', 'test', 'whole_train')
        assert arg.save_weight_strategy in ('best_eval', 'each_interval')
        self.arg = arg
        self.save_arg()
        self.load_data()
        self.load_model()
        self.load_optimizer()
        self.eval_acc_list = []
        
    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()
        if self.arg.phase in ('train', 'whole_train'):
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.train_batchsize,
                shuffle=True,   
                num_workers=self.arg.num_worker)
            
            self.data_loader['whole_train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.whole_train_feeder_args),
                batch_size=self.arg.train_batchsize,
                shuffle=True,
                num_workers=self.arg.num_worker
            )

            self.data_loader['eval'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.eval_feeder_args),
                batch_size=self.arg.eval_batchsize,
                shuffle=False,
                num_workers=self.arg.num_worker)
        else:
            self.data_loader['test'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batchsize,
                shuffle=False,
                num_workers=self.arg.num_worker)

    def load_model(self):
        if len(self.arg.devices) == 0:
            self.output_device = torch.device("cpu")
        else:
            self.output_device = self.arg.devices[0] if type(self.arg.devices) is list else self.arg.device
        
        meta_model = import_class(self.arg.model)
        self.model = meta_model(**self.arg.model_args).to(self.output_device)
        self.char_ctcLoss = nn.CTCLoss(blank=0).to(self.output_device)
        self.word_clsLloss = nn.CrossEntropyLoss().to(self.output_device)
        self.length_reg_loss = nn.MSELoss().to(self.output_device)
        if len(self.arg.devices) == 0:
            self.seq_cls_loss = LipSeqLoss(iscuda=False)
        else:
            self.seq_cls_loss = LipSeqLoss(iscuda=True, device=self.output_device).to(self.output_device)
        
        if self.arg.weights:
            self.arg.weights = gc.project_root_path+self.arg.weights
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)
            
            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)
        
        # multi GPU serving
        if type(self.arg.devices) is list:
            if len(self.arg.devices) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.devices,
                    output_device=self.output_device)
            
    def load_optimizer(self):
        if self.arg.optimizer.lower() == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer.lower() == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError('You need to specify optimizer between Adam and SGD')

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.is_print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        self.arg.work_dir = gc.project_root_path+self.arg.work_dir
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer.lower() == 'sgd' or self.arg.optimizer.lower() == 'adam':
            lr = self.arg.base_lr * (
                    0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def get_char_label(self, word_label_idx):
        word_label = gc.idx2word_codebook[word_label_idx]
        char_label = []
        for char in word_label:
            char_label += [gc.char2idx_codebook[char]]
        char_label = np.array(char_label)   
        return char_label 

        
    def train_eval(self, epoch, mode='train', save_model=False):
        assert mode in ('train', 'eval', 'whole_train')
        if mode in ('train', 'whole_train'):
            self.model.train()
        else:
            self.model.eval()
        
        self.print_log('{} epoch: {}'.format(mode, epoch + 1))
        loader = self.data_loader[mode]
        if self.arg.optimizer.lower() == 'sgd':
            lr = self.adjust_learning_rate(epoch)
        else:
            lr = 'adam optimizer setting'
        loss_value = []
        gt_label = []
        preds_label = []
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        loss_value = []
        loss_type = dict(loss_word=[], loss_char=[], loss_reg=[], loss_seq=[])
        for batch_idx, sample_batched in enumerate(loader):
            input_data = Variable(sample_batched['volume']).to(self.output_device)
            labels = Variable(sample_batched['label'])
            length = Variable(sample_batched['length']).to(self.output_device)
            
            logit_char, logit_word, logit_word_, reg_length = self.model(input_data)
            Tc, Nc, _ = logit_char.shape
            # prepare the loss params
            labels = labels.squeeze()
            targets = []
            targets_length = []
            for each_label in labels:
                char_label = self.get_char_label(each_label.cpu().data.numpy().tolist())
                targets.extend(char_label)
                targets_length.append(len(char_label))
            targets = torch.tensor(np.array(targets)).to(self.output_device)
            targets += 1 # target vary from 1 to cls_number = 431, while 0 represent the blank
                        
            targets_length = torch.tensor(np.array(targets_length)).to(self.output_device)
            input_lengths = torch.tensor(np.array([Tc]*Nc)).to(self.output_device)
            logit_char = logit_char.log_softmax(2)

            lossv_char = self.char_ctcLoss(logit_char, targets, input_lengths, targets_length)
            # ctc loss for char

            labels = labels.to(self.output_device)
            lossv_word = self.word_clsLloss(logit_word, labels)
            # word cls loss
            lossv_reglength = self.length_reg_loss(reg_length, targets_length.float())
            # reg length loss
            labels_ = labels.unsqueeze(-1)
            lossv_seq = self.seq_cls_loss(logit_word_, length, labels_)
            
            whole_tradeoff = 1+sum([v for k, v in self.arg.trade_off_args.items()])

            lossv = (lossv_char, lossv_reglength, lossv_seq)
            loss_type['loss_word'] += [lossv_word.data.item()]
            loss_type['loss_char'] += [lossv_char.data.item()]
            loss_type['loss_reg'] += [lossv_reglength.data.item()]
            loss_type['loss_seq'] += [lossv_seq.data.item()]
            loss_total = lossv_word+ sum([v * lossv[ind] for ind, (k, v) in enumerate(self.arg.trade_off_args.items())])
            loss_total /= whole_tradeoff

            if mode in ('train', 'whole_train'):
                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

            loss_value.append(loss_total.data.item())
            pred = torch.argmax(F.softmax(logit_word, dim=1), dim=1)
            pred = pred.cpu().data.numpy().tolist()
            label = labels.cpu().data.numpy().tolist()

            gt_label.extend(label)
            preds_label.extend(pred)
            
            timer['model'] += self.split_time()
            # statistics
            if batch_idx % self.arg.log_interval == 0:
                self.print_log(
                    '\tBatch({}/{}) done. Loss: {:.4f}  lr:{}'.format(
                        batch_idx, len(loader), loss_total.data.item(), lr))
                if not self.arg.is_detail_log:
                    continue
                self.print_log(
                    '\t Loss detail: loss_word: {:.4f}, loss_char: {:.4f}, loss_reg:{:.4f}, loss_seq:{:.4f}'.format(
                        np.mean(loss_type['loss_word']),
                        np.mean(loss_type['loss_char']),
                        np.mean(loss_type['loss_reg']),
                        np.mean(loss_type['loss_seq'])
                    )
                )
            timer['statistics'] += self.split_time()
        
        # statistics of time consumption and loss
        # metric calculate
        preds_label = np.array(preds_label)
        gt_label = np.array(gt_label)
        acc = np.sum(preds_label == gt_label) / len(preds_label)
        if mode == 'eval':
            self.eval_acc_list.append(acc)
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        
        print('\033[1;31;40m')
        print(' ===== LOOK HERE =====')
        self.print_log(
            '\tMean {} loss: {:.4f}. acc: {}'.format(mode, np.mean(loss_value), acc))
        if self.arg.is_detail_log:
            self.print_log(
                '\t Mean {} loss detail: loss_word: {:.4f}, loss_char:{: .4f}, loss_reg: {:.4f}, loss_seq:{:.4f}'.format(
                    mode,
                    np.mean(loss_type['loss_word']),
                    np.mean(loss_type['loss_char']),
                    np.mean(loss_type['loss_reg']),
                    np.mean(loss_type['loss_seq'])
                )
            )
            self.print_log(
                '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                    **proportion))
        print(' ===== LOOK HERE =====')
        print('\033[0m')

        if save_model:
            model_path = '{}/epoch{}_model_acc_{}.pt'.format(self.arg.work_dir,
                                                        epoch + 1,
                                                        acc)
            if self.arg.save_weight_strategy == 'best_eval':
                if acc == max(self.eval_acc_list) and acc >= self.arg.best_save_theshold:
                    torch.save(self.model.state_dict(), model_path)
            else:
                torch.save(self.model.state_dict(), model_path)

    def test_predict(self):
        self.model.eval()
        # generate the csv prediction files, at first we need to verify the eval accuracy
        gt_label = []
        preds_label = []
        for batch_idx, sample_batched in enumerate(self.data_loader['eval']):
            input_data = Variable(sample_batched['volume']).to(self.output_device)
            labels = Variable(sample_batched['label']).to(self.output_device)
            length = Variable(sample_batched['length']).to(self.output_device)
            _, logit_word, _,_ = self.model(input_data)
            pred = torch.argmax(F.softmax(logit_word, dim=1), dim=1)
            pred = pred.cpu().data.numpy().tolist()
            labels = labels.cpu().data.numpy().tolist()
            gt_label.extend(labels)
            preds_label.extend(pred)
        
        preds_label = np.array(preds_label)[:, np.newaxis]
        gt_label = np.array(gt_label)
        acc = np.sum(preds_label == gt_label) / len(preds_label)
        print('Eval accuracy is {}, check if you load the right weights!'.format(acc))
        print('Begin to generate the test csv files')
        preds = []
        filenames = []
        loader = self.data_loader['test']
        for batch_idx, sample_batched in enumerate(loader):
            print_toolbar(batch_idx * 1.0 / len(loader),
                '({:>5}/{:<5})'.format(
                    batch_idx + 1, len(loader)))
            input_data = Variable(sample_batched['volume']).to(self.output_device)
            hashname = sample_batched['hashname'][0]
            _, logit_word, _,_ = self.model(input_data)
            pred = torch.argmax(F.softmax(logit_word, dim=1), dim=1)
            pred = pred.cpu().data.numpy().tolist()[0]
            
            preds += [gc.idx2word_codebook[pred]]
            filenames += [hashname]
        end_toolbar()
        m = np.array(list(zip(filenames, preds)))
        df = pd.DataFrame(m)
        csv_path = self.arg.work_dir+'csv_result/'
        if not os.path.exists(csv_path):
            os.makedirs(csv_path)
        df.to_csv("{}/lipnet_acc_{}.csv".format(csv_path, acc), 
                                                header=False, 
                                                index_label=False, 
                                                index=False)
        print('save csv at {}'.format(csv_path))

    def start(self):
        if self.arg.phase in ('train', 'whole_train'):
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                eval_model = ((epoch + 1) % self.arg.eval_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)
                save_model = ((epoch+1) % self.arg.save_interval == 0) or (
                        epoch + 1 == self.arg.num_epoch)
                self.train_eval(epoch=epoch, mode=self.arg.phase, save_model=False)
                if eval_model:
                    with torch.no_grad():
                        if self.arg.save_weight_strategy == 'best_eval':
                            self.train_eval(epoch=epoch, mode='eval', save_model=True)
                        else:
                            self.train_eval(epoch=epoch, mode='eval', save_model=save_model)
        else:
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.is_print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            with torch.no_grad():
                self.test_predict()
            print('Done.\n')
            
    

if __name__ == '__main__':
    from codes.utils.parser import load_args
    arg = load_args()
    processor = Processor(arg)
    processor.print_log('Begin! here we go')
    processor.start()