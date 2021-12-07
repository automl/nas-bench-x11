# This is the code of nas-bench-nlp based on:
# N. Klyuchnikov et al. 2020, “NAS-Bench-NLP: 
# Neural Architecture Search Benchmark for Natural Language Processing", 
# arXiv preprint arXiv:1705.10823.

import sys
import os
import torch.utils.data
import numpy as np
import math
import time
from argparse import Namespace

from nasbenchnlp import data
from nasbenchnlp.utils import batchify
from nasbenchnlp.model import AWDRNNModel
from nasbenchnlp.train import train, evaluate
from nasbenchnlp.splitcross import SplitCrossEntropyLoss

#TODO: Save all train/val/test loss for each architectures of each algorithms.

def main_one_model_train(recipe):
    args = Namespace(data='nasbench_nlp/data/ptb',
                     recepie_id=0,
                     cuda=True,
                     batch_size=20,
                     model='CustomRNN',
                     emsize=400,
                     nhid=600,
                     nlayers=3,
                     dropout=0.4,
                     dropouth=0.25,
                     dropouti=0.4,
                     dropoute=0.1,
                     wdrop=0.5,
                     tied=True,
                     bptt=70,
                     lr=1e-3,
                     wdecay=1.2e-6,
                     epochs=3,
                     alpha=2,
                     beta=1,
                     log_interval=200,
                     clip=0.25,
                     eval_batch_size = 50,
                     recepie=recipe)
    
    corpus = data.Corpus(args.data)
    cuda = 'cuda'

    train_data = batchify(corpus.train, args.batch_size, args, cuda)
    train_eval_data = batchify(corpus.train, args.eval_batch_size, args, cuda)
    val_data = batchify(corpus.valid, args.eval_batch_size, args, cuda)
    test_data = batchify(corpus.test, args.eval_batch_size, args, cuda)
    
    ntokens = len(corpus.dictionary)
    
    custom_model = AWDRNNModel(args.model, 
                               ntokens, 
                               args.emsize, 
                               args.nhid, 
                               args.nlayers, 
                               args.dropout, 
                               args.dropouth, 
                               args.dropouti, 
                               args.dropoute, 
                               args.wdrop, 
                               args.tied,
                               args.recepie,
                               verbose=False)

    criterion = SplitCrossEntropyLoss(args.emsize, splits=[], verbose=False)
    
    if args.cuda:
        custom_model = custom_model.to(cuda)
        criterion = criterion.to(cuda)

    params = list(custom_model.parameters()) + list(criterion.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr, weight_decay=args.wdecay)

    train_losses = []
    val_losses = []
    test_losses = []
    wall_times = []

    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train(custom_model, optimizer, params, criterion, train_data, args, epoch)
        epoch_end_time = time.time()
        train_loss = evaluate(custom_model, criterion, train_eval_data, args.eval_batch_size, args)
        val_loss = evaluate(custom_model, criterion, val_data, args.eval_batch_size, args)
        test_loss = evaluate(custom_model, criterion, test_data, args.eval_batch_size, args)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s |\n| train loss {:5.2f} | '
            'train ppl {:8.2f} | train bpw {:8.3f} |\n| valid loss {:5.2f} | '
            'valid ppl {:8.2f} | valid bpw {:8.3f} |\n| test loss {:5.2f} | '
            'test ppl {:8.2f} | test bpw {:8.3f} |'.format(
          epoch, (epoch_end_time - epoch_start_time),
                train_loss, math.exp(train_loss), train_loss / math.log(2),
                val_loss, math.exp(val_loss), val_loss / math.log(2),
            test_loss, math.exp(test_loss), test_loss / math.log(2)))
        print('-' * 89)

        wall_times.append(epoch_end_time - epoch_start_time)

        if np.isnan(np.array([train_loss])) or np.isnan(np.array([val_loss])) or np.isnan(np.array([test_loss])):
            train_losses.append(6.5)
            val_losses.append(6.5)
            train_losses.append(6.5)
        else:
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            test_losses.append(test_loss)

    return train_losses, val_losses, test_losses
