""" Synaplexus Trainer Script 
"""
import os
import argparse

def parse_cmd_line_options():
    """
    """
    # Base argument parser for common arguments
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument('--model'       , type=str , help='DNN Model name. e.g. VGG.')
    base_parser.add_argument('--model-param' , type=str , help='Model Specific Parameters.')
    base_parser.add_argument('--input-size'  , type=int , help='Input Image Size.')
    base_parser.add_argument('--data-format' , type=str , help='Data Format. Either NCHW or NHWC.')
    base_parser.add_argument('--log-subdir'  , type=str , help='Logs Directory')
    base_parser.add_argument('--dataset'     , type=str , help='Target Dataset name.')

    arg_parser = argparse.ArgumentParser(prog='tf_dnn', usage='tf_dnn <subcommand> [args]', 
                                         description='Tensorflow Deep Learning Module.')
    subparsers = arg_parser.add_subparsers(title='Sub-Commands', dest='subcmd')
 
    train_parser = subparsers.add_parser('train', parents=[base_parser], help='Train a DNN.')
    train_parser.add_argument('--lr'          , type=float , help='Learning Rate.')
    train_parser.add_argument('--wd'          , type=float , help='L2 Regularization Parameter or Weight Decay.')
    train_parser.add_argument('--optimizer'   , type=str   , help='Optimizer for parameter update. e.g. adam, sgd')
    train_parser.add_argument('--fp16'        , type=int   , help='Use fp16 for the entire model parameters.')
    train_parser.add_argument('--data-aug'    , type=int   , help='Use data-augmentation to extend the dataset.')
    train_parser.add_argument('--batch-size'  , type=int   , help='Training mini-batch size.')
    train_parser.add_argument('--lr-step'     , type=int   , help='Learning rate decay step in epochs.')
    train_parser.add_argument('--lr-decay'    , type=float , help='Learning rate decay rate.')
    train_parser.add_argument('--num-epoch'   , type=int   , help='Number of epochs for the training process.')
    train_parser.add_argument('--begin-epoch' , type=int   , help='Epoch ID of from which the training will start. Useful for training resume.')

    deploy_parser = subparsers.add_parser('deploy', parents=[base_parser], help='Import a pretrained model for inference.')
    deploy_parser.add_argument('--checkpoint', type=int, help='Checkpoint ID', default=0)

    eval_parser = subparsers.add_parser('eval', parents=[base_parser], help='Import a pretrained model for inference.')
    eval_parser.add_argument('--checkpoint', type=int, help='Checkpoint ID', default=0)

    train_parser.set_defaults(
        lr          = 1e-3,
        wd          = 0,
        optimizer   = 'Adam',
        fp16        = 0,
        data_aug    = 0,
        batch_size  = 128,
        lr_step     = 10000,
        lr_decay    = 0,
        begin_epoch = 0,
        num_epoch   = 1
    )

    args = arg_parser.parse_args()
    return args
