import csv
import math
import time
import torch
import parser
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as Variable

import os
from util import *

# Define the training function
def train(model, criterion, tr_batcher, epoch, writer, logger, args ):
    num_batches = tr_batcher.num_batches
    total_loss = 0
    start_time = time.time()
    all_losses = []
    params = list(model.parameters())
    model.train()
    optimizer = torch.optim.SGD( params, args.lr, momentum = args.momentum, weight_decay = args.weight_decay )

    for batch_idx, (data, targets, lens) in enumerate(tr_batcher.next()):
        bsz = data.size(0)

        logits, attn_weights, sent_emb = model( data,  lens)  # bsz x max_sent x nclass

        # mean_attn_weights = torch.div(torch.sum(attn_weights, dim=1), torch.Tensor([args.aspects]).to(args.gpu_id))
        # l1 = torch.sum(mean_attn_weights)
        loss = criterion(logits, targets )# argument2 should be float tensor for some reason
        # I = torch.eye(attn_weights.size(1)).to(attn_weights.device)
        # AA_t_I = torch.bmm(attn_weights, attn_weights.transpose(2, 1)) - I

        # disagreement regularization on attended positions
        # AA_t = torch.bmm(attn_weights, attn_weights.transpose(2, 1))

        # disagreement regularization on sentence embeddings
        writer.add_scalar('train/loss', loss, (epoch-1)*num_batches+batch_idx) #global_step
        writer.add_scalar('train/lr', args.lr,(epoch-1)*num_batches+batch_idx)
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping in case of gradient explosion
        torch.nn.utils.clip_grad_norm_( model.parameters(), args.clip )

        optimizer.step()

        total_loss += loss.data
        all_losses.append( loss.data )
        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = total_loss / args.log_interval
            elapsed = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.4f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} |'.format(
                epoch, batch_idx, tr_batcher.num_batches, args.lr,
                elapsed * 1000 / args.log_interval, cur_loss ) )

            total_loss = 0
            start_time = time.time()
    return np.sum( all_losses )/len(all_losses)

    f.close()
# Define evaluation function
def evaluate( model, criterion, ev_batcher, epoch, writer, logger, args, mode='EVAL' ):
    num_batches = ev_batcher.num_batches
    total_loss = 0
    epoch_acc = []
    epoch_pre = []
    epoch_rec = []
    epoch_f1 = []
    epoch_preds = []
    epoch_targets =  []
    m = Metrics(args.nclass, list(range(args.nclass)))
    total_loss = 0.0
    model.eval()
    for batch_idx, (data, targets, lens) in enumerate(ev_batcher.next()):
        logits, attn_weights, sent_emb = model( data,  lens)  # bsz x max_sent x nclass
        loss = criterion(logits, targets )  #argument2 should be float tensor for some reason

        total_loss += loss.data
        _ , pred = logits.topk( 1 , 1)  # index of the correct class

        if len(epoch_targets) > 0:
            epoch_targets = np.concatenate([targets.cpu().numpy(), epoch_targets])
            epoch_preds = np.concatenate([pred.cpu().numpy(), epoch_preds])
        else:
            epoch_targets = targets.cpu().numpy()
            epoch_preds = pred.cpu().numpy()

        acc = m.accuracy(targets.cpu(), pred.cpu())
        pre = m.precision(targets.cpu(), pred.cpu(), average='binary')
        rec = m.recall(targets.cpu(), pred.cpu(), average='binary')
        f1 =  m.f1(targets.cpu(), pred.cpu(), average='binary')

        if mode=='EVAL':
            writer.add_scalar('eval/loss', loss.data, (epoch-1)*num_batches+batch_idx)
            writer.add_scalar('eval/accuracy', acc, (epoch-1)*num_batches+batch_idx)
            writer.add_scalar('eval/precision', pre, (epoch-1)*num_batches+batch_idx)
            writer.add_scalar('eval/recall', rec, (epoch-1)*num_batches+batch_idx)
            writer.add_scalar('eval/f1_score', f1, (epoch-1)*num_batches+batch_idx)

        epoch_acc.append(acc)
        epoch_pre.append(pre)
        epoch_rec.append(rec)
        epoch_f1.append(f1)

    # Compute Precision, Recall, F1, and Accuracy
    logger.info('Measure on this dataset')
    logger.info('Precision: {}'.format(np.mean( epoch_pre )))
    logger.info('Recall: {}'.format(np.mean( epoch_rec )))
    logger.info('F1: {}'.format(np.mean( epoch_f1 )))
    logger.info('Acc: {}'.format(np.mean( epoch_acc )))

    return total_loss/ev_batcher.num_batches, np.mean(epoch_pre), np.mean(epoch_rec), np.mean(epoch_f1), np.mean( epoch_acc ), epoch_targets, epoch_preds
