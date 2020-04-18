import csv
import math
import time
import json
import torch
import parser
import argparse
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as Variable
from tensorboardX import SummaryWriter

# from data.data_multiclass import *
from data import *
from model import *
from util import *
from train import *

import random
random.seed(3242)

def build_model(args, embeddings=None):
    model = CNNMaxPool(args)
    if args.emb:
        model.set_embedding(embeddings) #TODO:
    return model

def main(args):
    if args.cuda:
        torch.cuda.set_device(args.gpu_id)

    log_level = logging.INFO
    # formatter_str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if not args.logfile:
        logging.basicConfig(stream=sys.stdout, format=formatter_str, level=log_level)
    else:
        logging.basicConfig(filename=args.logfile, level=log_level)

    logger = logging.getLogger("MIL")

    logger.info('Current device: '.format(torch.cuda.current_device()))

    logger.info('Building corpus....')
    corpus = Corpus(args)

    #concat all splits
    data = corpus.train
    labels = corpus.train_labels
    lens  = corpus.train_lens

    embeddings=None
    if args.emb:
        logger.info('Loading word embeddings, this could take a while...')
        embeddings = get_embeddings(args.emb_path, corpus.dictionary, args.emb_dim)


    prec = []
    reca = []
    f1sc = []
    accu = []

    orig_lr = args.lr
    best_val_loss = None

    for fold in range(args.folds):
        print('Training for fold {}'.format(fold))
        writer = SummaryWriter()

        cv = CrossValidationSplits(data, labels, lens)
        logger.info('************Training for fold: {} **************'.format(fold))
        model = build_model(args, embeddings)  # build a new model every time

        criterion = nn.CrossEntropyLoss()

        if args.cuda:
            model.cuda()
            criterion.cuda()

        args.lr = orig_lr
        logger.info('Start training...')
        lr = float(args.lr)
        patience_cnt = 0
        train_losses = []
        t = []

        # main training loop
        for epoch in range( 1, args.epochs + 1 ):
            print('Training for epoch: {}/{}'.format(epoch, args.epochs))
            epoch_start_time = time.time()
            tr_batcher = Batchify(cv.train, cv.train_labels, cv.train_lens, bsz=args.batch_size, cuda=args.cuda)
            val_batcher = Batchify(cv.val, cv.val_labels, cv.val_lens, bsz=args.batch_size, cuda=args.cuda)
            train_losses.append( train( model, criterion, tr_batcher, epoch, writer, logger, args ) )
            val_loss, pre, rec, f1, acc, targets, preds = evaluate( model, criterion, val_batcher, epoch, writer, logger, args )
            # Save the best model and Anneal the learning rate.
            if not best_val_loss or val_loss < best_val_loss:  # save best model accross folds
                best_val_loss = val_loss
                patience_cnt = 0
                with open( args.save, 'wb+' ) as f:
                    torch.save( model , f )
            else:
                patience_cnt += 1
                args.lr /= 2.0
                if patience_cnt == args.patience:  #stop training if val loss hasn't increased for 5 epochs
                    print('Early stopping after {} epochs.'.format(epoch))
                    break;


    print('Best Model')
    with open( args.save, 'rb+' ) as f:
        model = torch.load(f)  #load best model

    test_batcher = Batchify(corpus.test, corpus.test_labels, corpus.test_lens, bsz=args.batch_size, cuda=args.cuda)
    test_loss, pre, rec, f1, acc, targets, preds = evaluate( model, criterion, test_batcher, epoch, writer, logger, args, mode='TEST')
    report  = classification_report(targets, preds, output_dict=True)
    report['accuracy'] = accuracy_score(targets, preds)
    if not os.path.exists(os.path.join(args.stats_dir, args.experiment+'/TEST')):
        os.makedirs(os.path.join(args.stats_dir,  args.experiment+'/TEST'))
    experiment_path = os.path.join(args.stats_dir, args.experiment+'/TEST')
    with open(os.path.join(experiment_path, 'best_model.json'.format(fold)), 'w+') as f:
        json.dump(report, f)


    test_batcher = Batchify(corpus.test, corpus.test_labels, corpus.test_lens, bsz=args.batch_size, cuda=args.cuda)
    test_loss, pre, rec, f1, acc, targets, preds = evaluate( model, criterion, test_batcher, epoch, writer, logger, args, mode='TEST')
    report  = classification_report(targets, preds, output_dict=True)
    report['accuracy'] = accuracy_score(targets, preds)
    if not os.path.exists(os.path.join(args.stats_dir, args.experiment+'/TEST')):
        os.makedirs(os.path.join(args.stats_dir,  args.experiment+'/TEST'))
    experiment_path = os.path.join(args.stats_dir, args.experiment+'/TEST')
    with open(os.path.join(experiment_path, '_fold_{}_test.json'.format(fold)), 'w+') as f:
        json.dump(report, f)

    writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')

    # train
    parser.add_argument('--epochs', type=int, default=50, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--eval_size', type=int, default=32, help='Evaluation batch size')
    parser.add_argument('--seed', type=int, default=3242, help='random seed')
    parser.add_argument('--pretrained', type=str, default='', help='whether start from pretrained model')
    parser.add_argument('--cuda', default=True, action='store_true', help='use CUDA')
    parser.add_argument('--log_interval', type=int, default=10, help='report interval')
    parser.add_argument('--logfile', type=str, default='./.log', help='log to file')
    parser.add_argument('--save', type=str, default='/home/sudo777/ASONAM/output_yelp/cnn.pt', help='path to save the final model')
    parser.add_argument('--lr', type=float, default=0.05, help='Evaluation batch size')
    parser.add_argument('--momentum', type=float, default=0.9, help='sgd with momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight_decay')
    parser.add_argument('--folds', type=int, default=3, help='number of cross validation folds')
    parser.add_argument('--gpu_id', type=str, default=1, help='')
    parser.add_argument('--patience', type=int, default=3, help='patience for early stopping')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping')


    #data
    parser.add_argument('--data_dir', type=str, default='/home/sudo777/ASONAM/data/yelp_p/', help='Location of the data corpus')
    parser.add_argument('--emb_path', type=str, default='/home/sudo777/ASONAM/data/yelp_p/word2vec.100d.54k.w2v', help='path to the embedding file')
    parser.add_argument('--from_emb', default=True, action='store_true', help='Whether build vocab from embedding file')
    parser.add_argument('--vocab_size', type=int, default=50000, help='vocab size, only used when --from_emb is False. Vocabulary is built from data.')
    parser.add_argument('--stop_words', type=str, default='/home/sudo777/mercury/data_autogsr/en_stopwords.txt', help='Location of the data corpus')
    parser.add_argument('--emb', default=True, action='store_true', help='Whether to load pretrained word embddings')

    #stats
    parser.add_argument('--experiment', type=str, default='YELP-P/analyses/CNN', help='Location of the data corpus')
    parser.add_argument('--stats_dir', type=str, default='/home/sudo777/ASONAM/stats', help='location to save classification reports')
    #model
    parser.add_argument('--emb_dim', type=int, default=100, help='Location of the data corpus')
    parser.add_argument('--max_sent', type=int, default=15, help='Max number of sents in a doc')
    parser.add_argument('--max_word', type=int, default=250, help='Max number of words in a sent')
    parser.add_argument('--num_kernels', type=int, default=3, help='Evaluation batch size')
    parser.add_argument('--num_filters', type=int, default=100, help='random seed')
    parser.add_argument('--dropout', type=float, default=0.4, help='whether start from pretrained model')
    parser.add_argument('--nclass', type=int, default=8, help='use CUDA')
    parser.add_argument('--mlp_nhid', type=int, default=512, help='Number of hidden units in the sentence classifier') # prev 128
    parser.add_argument('--lstm_layer', type=int, default=1, help='Evaluation batch size')
    parser.add_argument('--lstm_hidden', type=int, default=50, help='Evaluation batch size')
    parser.add_argument('--n_layers', type=int, default=1, help='Number of GRU stacking layers')
    parser.add_argument('--bidirectional', default=True, action='store_true', help='Evaluation batch size')
    parser.add_argument('--alpha', type=float,  default=0.5, help='constant controlling instance ratio' )
    parser.add_argument('--beta', type=float,  default=0.5, help='constant controlling instance ratio' )
    parser.add_argument('--m0', type=float,  default=0.5, help='margin for instance loss' )
    parser.add_argument('--p0', type=float,  default=0.6, help='instance loss constant' )
    parser.add_argument('--K', type=int,  default=3, help='Number of sentences to select' )
    parser.add_argument('--da', type=int, default=350, help='Hidden dimension for self attention')
    parser.add_argument('--aspects', type=int, default=5, help='Number of aspects')
    parser.add_argument('--d_ff', type=int, default=1024, help='Hidden dimension of Feedforward layers')
    parser.add_argument('--u_w_dim', type=int, default=32, help='Dimension of the sentence-level context vector')

    args = parser.parse_args()
    main(args)
