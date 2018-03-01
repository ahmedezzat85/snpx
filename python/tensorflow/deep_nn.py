"""
"""
import os
import json
from datetime import datetime

import utils
from tf_dnn import TFClassifier
from mv_dataset import TFDatasetReader
from arg_parser import parse_cmd_line_options

_CUR_DIR       = os.path.dirname(__file__)
_LOGS_ROOT_DIR = os.path.join(_CUR_DIR, '..', 'logs')
_MODEL_BIN_DIR = os.path.join(_CUR_DIR, '..', 'bin')

class DeepNN(object):
    """
    """
    def __init__(self, args):
        self.flags    = args
        self.logs_dir = os.path.join(_LOGS_ROOT_DIR, args.model_name, args.log_subdir)
        self.bin_dir  = os.path.join(_MODEL_BIN_DIR, args.model_name, args.log_subdir)
        model_param   = args.model_param.strip().split(',') if args.model_param is not None else []
        self.dataset = TFDatasetReader(image_size=args.input_size)
        self.module  = TFClassifier(self.dataset, args.model_name, args.data_format, self.logs_dir, *model_param)

    def train(self):        
        # Pack the hyper-parameters
        hp_dict = {'batch_size': self.flags.batch_size, 'optimizer': self.flags.optimizer,
                    'lr': self.flags.lr, 'wd': self.flags.wd, 'lr_decay': self.flags.lr_decay,
                    'lr_decay_epochs': self.flags.lr_step, 'data_aug': self.flags.data_aug}
        hp = utils.DictToAttrs(hp_dict)
        
        # Train the model
        self.module.train(hp, self.flags.num_epoch, self.flags.begin_epoch)

    def evaluate(self):
        """ """
        self.module.evaluate(self.flags.checkpoint)

    def deploy(self):
        """ """
        self.module.deploy(self.bin_dir, self.flags.input_size, self.dataset.num_classes, self.flags.checkpoint)

def main():
    """ """
    args = parse_cmd_line_options()
    cmd = args.subcmd.lower()

    dnn = DeepNN(args)
    if   cmd == 'train' : dnn.train()
    elif cmd == 'deploy': dnn.deploy()
    elif cmd == 'eval'  : dnn.evaluate()
    else:
        raise ValueError('Unknown sub-command %s', cmd)

if __name__ == '__main__':
    main()
