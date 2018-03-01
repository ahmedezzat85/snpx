from __future__ import absolute_import

import os
import sys
import cv2
from time import time
from datetime import datetime
from importlib import import_module

import numpy as np
import tensorflow as tf

import utils

class TFClassifier(object):
    """ Abtraction of TensorFlow functionality.
    """
    def __init__(self, dataset, model_name, data_format='NHWC', logs_dir=None, *model_params):

        # Display Tensorflow Version
        print ('Tensorflow Version:   ', tf.__version__)
        tf.logging.set_verbosity(tf.logging.INFO)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Parameter initializations
        self.loss         = None
        self.dtype        = tf.float32
        self.logger       = None
        self.log_dir      = logs_dir
        self.tf_sess      = None
        self.dataset      = dataset
        self.train_op     = None
        self.training     = None
        self.base_tick    = time()
        self.model_name   = model_name
        self.summary_op   = None
        self.accuracy_op  = None
        self.data_format  = data_format
        self.global_step  = None
        self.summary_list = []

        # Get the neural network model function
        net_module = import_module('models.' + model_name)
        self.model = net_module.TFModel(self.dtype, data_format, dataset.num_classes, *model_params)
        
        self.chkpt_dir = os.path.join(self.log_dir, 'chkpt')
        utils.create_dir(self.chkpt_dir)
        self.chkpt_prfx = os.path.join(self.chkpt_dir, 'CHKPT')

    def tick(self):
        return time() - self.base_tick

    def _dump_hyperparameters(self, begin_epoch):
        """ """
        model_param =[]
        for k, v in self.model.config.items():
            item = ['**'+k+'**', str(v)]
            model_param.append(item)
        hp = [
            ['**Batch Size**', str(self.hp.batch_size)],
            ['**Optimizer**', self.hp.optimizer], 
            ['**Learning Rate**', str(self.hp.lr)], 
            ['**Weight Decay**', str(self.hp.wd)], 
            ['**LR Decay**', str(self.hp.lr_decay)],
            ['**Decay Epochs**', str(self.hp.lr_decay_epochs)]]
        summ_op = tf.summary.merge(
                    [tf.summary.text(self.model_name + '/HyperParameters', tf.convert_to_tensor(hp)),
                    tf.summary.text(self.model_name + '/Dataset', tf.convert_to_tensor(self.dataset.name)),
                    tf.summary.text(self.model_name + '/Model', tf.convert_to_tensor(model_param))])
        s = self.tf_sess.run(summ_op)
        self.tb_writer.add_summary(s, begin_epoch)
        self.tb_writer.flush()

    def _load_dataset(self, batch_size, training=True):
        """ """
        with tf.device('/cpu:0'):
            self.dataset.read(batch_size, training, self.data_format, training, self.dtype)
    
    def loss_fn(self, logits, labels):
        cross_entropy = tf.losses.softmax_cross_entropy(labels, logits)
        self.summary_list.append(tf.summary.scalar("Loss", cross_entropy))
        return cross_entropy

    def _create_train_op(self, logits, labels):
        """ """
        self.global_step = tf.train.get_or_create_global_step()
        # Get the optimizer
        if self.hp.lr_decay:
            decay_steps = int((self.dataset.dataset_sz / self.hp.batch_size) * self.hp.lr_decay_epochs)
            lr = tf.train.exponential_decay(self.hp.lr, self.global_step, decay_steps, self.hp.lr_decay, True)
        else:
            lr = self.hp.lr
        self.summary_list.append(tf.summary.scalar("Learning-Rate", lr))

        optmz = self.hp.optimizer.lower()
        if optmz == 'sgd':
            opt = tf.train.MomentumOptimizer(lr, momentum=0.9)
        elif optmz == 'adam':
            opt = tf.train.AdamOptimizer(lr)
        elif optmz == 'rmsprop':
            opt = tf.train.RMSPropOptimizer(lr, momentum=0.9, epsilon=1)
 
        # Compute the loss and the train_op
        self.loss = self.loss_fn(logits, labels)
        if self.hp.wd > 0:
            l2_loss = self.hp.wd * tf.add_n(
                [tf.nn.l2_loss(v) for v in tf.trainable_variables() 
                    if ('batch_normalization' not in v.name) and ('BatchNorm' not in v.name)])
            self.loss = self.loss + l2_loss

        update_ops  = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = opt.minimize(self.loss, self.global_step)
        self._create_accuracy_op(logits, labels)

    def _create_accuracy_op(self, predictions, labels):
        """ """
        top1_tensor  = tf.equal(tf.argmax(predictions, axis=1), tf.argmax(labels, axis=1))
        top5_tensor  = tf.nn.in_top_k(predictions, tf.argmax(labels, axis=1), 5)
        top1_op      = tf.multiply(tf.reduce_mean(tf.cast(top1_tensor, tf.float32)), 100.0)
        top5_op      = tf.multiply(tf.reduce_mean(tf.cast(top5_tensor, tf.float32)), 100.0)
        self.summary_list.append(tf.summary.scalar("Top-1 Train Acc", top1_op))
        self.summary_list.append(tf.summary.scalar("Top-5 Train Acc", top5_op))
        self.accuracy_op = [top1_op, top5_op]

    def _train_loop(self):
        """ """
        # Initialize the training Dataset Iterator
        self.tf_sess.run(self.dataset.train_init_op)

        epoch_start_time = self.tick()
        last_log_tick    = epoch_start_time
        last_step        = self.tf_sess.run(self.global_step)
        top1_acc = 0
        top5_acc = 0
        while True:
            try:
                feed_dict = {self.training: True}
                fetches   = [self.loss, self.train_op, self.accuracy_op, self.summary_op, self.global_step]
                loss, _, [top1, top5], s, step = self.tf_sess.run(fetches, feed_dict)

                self.tb_writer.add_summary(s, step)
                self.tb_writer.flush()
                elapsed = self.tick() - last_log_tick
                if elapsed >= self.log_freq:
                    speed = ((step - last_step)  * self.hp.batch_size) / elapsed
                    last_step = step
                    last_log_tick  = self.tick()
                    self.logger.info('(%.3f)Epoch[%d] Batch[%d]\tloss: %.3f\tspeed: %.3f samples/sec\ttrain_acc = %.2f', 
                                      self.tick(), self.epoch, step, loss, speed, top1)
            except tf.errors.OutOfRangeError:
                break
        self.saver.save(self.tf_sess, self.chkpt_prfx, self.epoch + 1)
        self.logger.info('Epoch Training Time = %.3f', self.tick() - epoch_start_time)
        self.logger.info('Epoch[%d] Top-1 Train Acc = %.2f%%', self.epoch, top1)
        self.logger.info('Epoch[%d] Top-5 Train Acc = %.2f%%', self.epoch, top5)

    def _eval_loop(self):
        """ """
        self.tf_sess.run(self.dataset.eval_init_op)
        epoch_start_time = self.tick()
        top1_acc = 0
        top5_acc = 0
        n = 0
        feed_dict = None if self.training is None else {self.training: False}
        while True:
            try:
                top1, top5 = self.tf_sess.run(self.accuracy_op, feed_dict)
                top1_acc += top1
                top5_acc += top5
                n += 1
            except tf.errors.OutOfRangeError:
                break

        top1_acc = top1_acc / n
        top5_acc = top5_acc / n
        self.logger.info('Validation Time = %.3f', self.tick() - epoch_start_time)
        return top1_acc, top5_acc

    def _log_accuracy(self, tag, top1_acc, top5_acc, step):
        top1_summ = tf.summary.Summary()
        top1_summ.value.add(simple_value=top1_acc, tag='Top-1 Acc' +'('+tag+')')
        top5_summ = tf.summary.Summary()
        top5_summ.value.add(simple_value=top5_acc, tag='Top-5 Acc' +'('+tag+')')
        self.tb_writer.add_summary(top1_summ, step)
        self.tb_writer.add_summary(top5_summ, step)
        self.tb_writer.flush()

    def create_tf_session(self):
        """ """
        # Session Configurations 
        config = tf.ConfigProto()
        config.intra_op_parallelism_threads = 4
        config.gpu_options.allow_growth = True # Very important to avoid OOM errors
        config.gpu_options.per_process_gpu_memory_fraction = 1.0 #0.4

        # Create and initialize a TF Session
        self.tf_sess = tf.Session(config=config)
        self.tf_sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    def train(self, hp, num_epoch, begin_epoch, log_freq_sec=1):
        """ """
        self.hp       = hp
        self.log_freq = log_freq_sec

        t_start = datetime.now()
        self.logger = utils.create_logger(self.model_name, os.path.join(self.log_dir, 'Train.log'))
        self.logger.info("Training Started at  : " + t_start.strftime("%Y-%m-%d %H:%M:%S"))

        with tf.Graph().as_default():
            # Load the dataset
            self._load_dataset(self.hp.batch_size)

            # Forward Propagation
            self.training = tf.placeholder(tf.bool, name='Train_Flag')
            logits, probs = self.model.forward(self.dataset.images, self.training)
        
            self._create_train_op(logits, self.dataset.labels)

            # Create a TF Session
            self.create_tf_session()

            # Create Tensorboard stuff
            self.summary_op = tf.summary.merge(self.summary_list)
            self.tb_writer  = tf.summary.FileWriter(self.log_dir, graph=self.tf_sess.graph)
            self._dump_hyperparameters(begin_epoch)

            if begin_epoch > 0:
                # Load the saved model from a checkpoint
                chkpt = self.chkpt_prfx + '-' + str(begin_epoch)
                self.logger.info("Loading Checkpoint " + chkpt)
                num_epoch   += begin_epoch
                self.saver = tf.train.Saver(max_to_keep=50)
                self.saver.restore(self.tf_sess, chkpt)
                self.tb_writer.reopen()
            else:
                self.model.weight_init(self.tf_sess)
                self.saver = tf.train.Saver(max_to_keep=200)

            # Training Loop
            for self.epoch in range(begin_epoch, num_epoch):
                # Training
                self._train_loop()

                # Validation
                top1_acc, top5_acc = self._eval_loop()
                self._log_accuracy('Validation', top1_acc, top5_acc, self.epoch)
                self.logger.info('Epoch[%d] Top-1 Val Acc = %.2f%%', self.epoch, top1_acc)
                self.logger.info('Epoch[%d] Top-5 Val Acc = %.2f%%', self.epoch, top5_acc)

            # Close and terminate
            self.tb_writer.close()
            self.tf_sess.close()

        self.logger.info("Training Finished at : " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self.logger.info("Total Training Time  : " + str(datetime.now() - t_start))

    def evaluate(self, chkpt_id=None):
        """ """
        self.logger = utils.create_logger(self.model_name, os.path.join(self.log_dir, 'Eval.log'))
        with tf.Graph().as_default():
            # Load the Evaluation Dataset
            self._load_dataset(1, training=False)

            # Forward Prop
            predictions, _ = self._forward_prop(self.dataset.images, self.dataset.num_classes, False)

            # Create a TF Session
            self.create_tf_session()
 
            # Load the saved model from a checkpoint
            if chkpt_id is None:
                chkpt_state = tf.train.get_checkpoint_state(self.model_dir)
                chkpt = chkpt_state.model_checkpoint_path
            else:
                chkpt = self.chkpt_prfx + '-' + str(chkpt_id)

            self.logger.info("Loading Checkpoint " + chkpt)
            tf_model = tf.train.Saver()
            tf_model.restore(self.tf_sess, chkpt)

            # Perform Model Evaluation
            self._create_accuracy_op(predictions, self.dataset.labels)
            top1_acc, top5_acc = self._eval_loop()
            self.logger.info('Top-1 Accuracy = %.2f%%', top1_acc)
            self.logger.info('Top-5 Accuracy = %.2f%%', top5_acc)

            self.tf_sess.close()
        return top1_acc, top5_acc

    def deploy(self, deploy_dir, img_size, num_classes, chkpt_id=-1):
        """ Save a model for Inference.

        Parameters
        ----------
        deploy_dir: path
        num_classes: int
        chkpt_id: integer
            ID for the checkpoint to load the model from.
            * **"0"** means get the latest checkpoint
            * **"-1"** means no checkpoint. The model weights are randomly initialized.
              This is suitable for testing a model before training it. 
        """
        utils.create_dir(deploy_dir)
        with tf.Graph().as_default() as g:
            in_shape = [1, 3, img_size, img_size] if self.data_format == 'NCHW' else [1, img_size, img_size, 3]
            input_image = tf.placeholder(self.dtype, in_shape, name='input')
            _, predictions = self.model.forward(input_image, is_training=False)

            # Load the Evaluation Dataset
            saver = tf.train.Saver(tf.global_variables())
            self.create_tf_session()
            self.tb_writer  = tf.summary.FileWriter(deploy_dir, graph=self.tf_sess.graph)
            if chkpt_id >= 0:
                if chkpt_id == 0:
                    chkpt_state = tf.train.get_checkpoint_state(self.chkpt_dir)
                    chkpt = chkpt_state.model_checkpoint_path
                else:
                    chkpt = self.chkpt_prfx + '-' + str(chkpt_id)
                saver.restore(self.tf_sess, chkpt)
    
            image = np.random.rand(*in_shape)
            self.tf_sess.run(predictions, {input_image: image})

            saver.save(self.tf_sess, os.path.join(deploy_dir, 'network'))
            tf.train.write_graph(self.tf_sess.graph_def, deploy_dir, self.model_name+'.pb', False)
            tf.train.write_graph(self.tf_sess.graph_def, deploy_dir, self.model_name+'.pbtxt')

            self.tb_writer.close()
            self.tf_sess.close()

