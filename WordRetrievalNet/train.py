#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-
from __future__ import print_function

import os
import gc
import time
import datetime
import torch
import shutil
import pathlib
import numpy as np
from PIL import Image
from pprint import pformat
from torch import nn
from torch.utils import data
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder

import argparse
from datetime import timedelta
import signal, sys

from config import pick_configuration
from datasets.dataset_loader import CustomDataSetRBox, build_embedding_descriptor
from model import WordRetrievalModel, ModelLoss
from utils import setup_logger, cal_recall_precison_f1, cal_map, cal_overlap
from predict import resize_img, get_boxes, adjust_ratio



    

class Trainer:
    def __init__(self, config):
        self.config = config
        self.config['trainer']['output_dir'] = os.path.join(str(pathlib.Path(os.path.abspath(__name__)).parent),
                                                            self.config['trainer']['output_dir'])
        self.data_cfg = self.config["data_cfg"]
        self.dataset_name = self.data_cfg['name']
        self.method_name = "{0}_{1}".format(self.config['arch']['backbone'], self.dataset_name)
        self.save_dir = os.path.join(self.config['trainer']['output_dir'], self.method_name)
        self.checkpoint_dir = os.path.join(self.save_dir, 'checkpoint')
        if self.config['trainer']['resume_checkpoint'] == '' and self.config['trainer']['finetune_checkpoint'] == '':
            shutil.rmtree(self.save_dir, ignore_errors=True)
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)

        self.global_step = 0
        self.start_epoch = 1

        self.tensorboard_enable = self.config['trainer']['tensorboard']
        self.epochs = self.config['trainer']['epochs']
        self.save_interval = self.config['trainer']['save_interval']
        self.show_images_interval = self.config['trainer']['show_images_interval']
        self.display_interval = self.config['trainer']['display_interval']
        if self.tensorboard_enable:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(self.save_dir)

        # setup logger
        self.logger = setup_logger(os.path.join(self.save_dir, 'train_log'))
        self.logger.info(pformat(self.config))

        # device
        torch.manual_seed(self.config['trainer']['seed'])  #Set random seed for cpu
        self.logger.info("Is CUDA available?? {}".format(torch.cuda.is_available()))
        if len(self.config['trainer']['gpus']) > 0 and torch.cuda.is_available():
            self.with_cuda = True
            torch.backends.cudnn.benchmark = True
            self.logger.info('Train with gpu {} & PyTorch {}'.format(self.config['trainer']['gpus'], torch.__version__))
            self.gpus = {i: item for i, item in enumerate(self.config['trainer']['gpus'])}
            self.device = torch.device("cuda:0")
            torch.cuda.manual_seed(self.config['trainer']['seed'])  #Set a random seed for the current gpu
            torch.cuda.manual_seed_all(self.config['trainer']['seed'])  #Set random seeds for all gpu
        else:
            # assert False, "GPU is not used"
            self.with_cuda = False
            self.logger.info('Train with cpu & PyTorch {}'.format(torch.__version__))
            self.device = torch.device("cpu")
        self.logger.info('Device {}'.format(self.device))

        # train data loader
        self.logger.info('Loading train data...')
        self.train_data_len = len(os.listdir(self.data_cfg["train_img_path"]))
        self.train_set = CustomDataSetRBox(self.data_cfg, max_img_length=self.config["trainer"]["input_size"],
                                           long_size=self.config['trainer']['long_size'])
        # self.embedding_size, self.words_embeddings = build_embedding_descriptor(self.train_set.unique_words)
        self.embedding_size = self.train_set.embedding_size
        self.words_embeddings = self.train_set.words_embeddings
        self.train_loader = data.DataLoader(self.train_set, batch_size=self.config["trainer"]["batch_size"],
                                            shuffle=True, num_workers=self.config["trainer"]["num_workers"],
                                            drop_last=False, pin_memory=True)
        self.train_loader_len = len(self.train_loader)
        self.logger.info('Train data has {0} samples, {1} in loader'.format(self.train_data_len, self.train_loader_len))

        # test data loader
        self.test_gt_path = self.train_set.val_gt_path#test_gt_path
        self.test_img_files = self.train_set.val_img_files#test_img_files
        self.test_gt_files = self.train_set.val_gt_files#test_gt_files
        self.test_words = self.train_set.val_words#test_words
        self.train_unique_words = self.train_set.train_unique_words
        self.label_encoder = LabelEncoder()

        # model
        self.logger.info('Loading model...')
        self.model = WordRetrievalModel(n_out=self.embedding_size, backbone=self.config["arch"]["backbone"],
                                        pre_trained=self.config["arch"]["pre_trained"])

        # loss function
        self.logger.info('Loading loss function...')
        self.criterion = ModelLoss(weight_cls=self.config["loss"]["weight_cls"],
                                   weight_angle=self.config["loss"]["weight_angle"],
                                   weight_diou=self.config["loss"]["weight_diou"],
                                   weight_embed=self.config["loss"]["weight_embed"])

        # optimizer and lr_scheduler
        self.logger.info('Loading optimizer and lr_scheduler...')
        self.lr = self.config["optimizer"]['args']['lr']
        self.lr_step = self.config["trainer"]["lr_step"]
        self.optimizer = self._initialize('optimizer', torch.optim, self.model.parameters())
        self.scheduler = self._initialize('lr_scheduler', torch.optim.lr_scheduler, self.optimizer)
        if self.config['trainer']['resume_checkpoint'] != '':
            self._load_checkpoint(self.config['trainer']['resume_checkpoint'], resume=True)
        elif self.config['trainer']['finetune_checkpoint'] != '':
            self._load_checkpoint(self.config['trainer']['finetune_checkpoint'], resume=False)

        # eval args
        self.cls_score_thresh = self.config['tester']['cls_score_thresh']
        self.bbox_nms_overlap = self.config['tester']['bbox_nms_overlap']
        self.query_nms_overlap = self.config['tester']['query_nms_overlap']
        self.overlap_thresh = 0.25
        self.metric = self.config['tester']['distance_metric']


        #Single machine with multiple cards
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            self.model = nn.DataParallel(self.model)
        self.model.to(self.device)


        self.metrics = {'precision': 0, 'recall': 0, 'hmean': 0, 'map': 0, 'mr': 0, 'train_loss': float('inf'), 'best_model': ''}
        
        self.epoch_result = {
            "epoch": -1,
            "train_loss": np.infty,
            "time": -1,
            "lr": np.infty,
        }
        
    def train(self):
        """ Full training logic """
        self._on_epoch_finish(self.start_epoch)
        self.logger.info('Start training...')
        for epoch in range(self.start_epoch, self.epochs + 1):
            # print(f"Training {epoch=}")
            gc.collect()
            torch.cuda.empty_cache()
            
            try:
                self.adjust_learning_rate(epoch)
                self.epoch_result = self._train_epoch(epoch)
                self._on_epoch_finish(epoch)
            except torch.cuda.OutOfMemoryError:
                self.logger.info("Crashing due to cuda running out of memory")
                self.logger.info(f"Memory usage during crash\n{torch.cuda.memory_summary()}")
                self._log_memory_usage()
                break
            except torch.cuda.CudaError:
                self._log_memory_usage()
        if self.tensorboard_enable:
            self.writer.close()
        if epoch < self.display_interval:
            self._on_epoch_finish(self.display_interval)
        self._on_train_finish()

    def _train_epoch(self, epoch):
        """ Training logic for an epoch """
        self.model.train()
        epoch_start, batch_start = time.time(), time.time()
        train_loss = 0.0
        lr = self.optimizer.param_groups[0]['lr']
        
        time_to_load_from_train_loader = time.time()                                                   

        for i, (img, gt_score, gt_geo, ignored_map, gt_embedding) in enumerate(self.train_loader):
            
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            lr = self.optimizer.param_groups[0]['lr']

            cur_batch_size = img.size()[0]
            img, gt_score, gt_geo, ignored_map, gt_embedding = img.to(self.device), gt_score.to(self.device), \
                                                               gt_geo.to(self.device), ignored_map.to(self.device), \
                                                               gt_embedding.to(self.device)

            (predict_score, predict_geo), predict_embedding = self.model(img)

            loss_all, loss_cls, loss_ang, loss_diou, loss_embed = self.criterion(
                gt_score, predict_score, gt_geo, predict_geo, gt_embedding, predict_embedding, ignored_map)

            # backward
            self.optimizer.zero_grad()
            
            loss_all.backward()
            
            self.optimizer.step()

            loss_all = loss_all.item()
            loss_cls, loss_ang, loss_diou = loss_cls.item(), loss_ang.item(), loss_diou.item()
            loss_embed = loss_embed.item()
            train_loss += loss_all
            
            
            time_to_load_from_train_loader = time.time()

            if i % self.display_interval == 0 or i == self.train_loader_len - 1:
                batch_time = time.time() - batch_start
                self.logger.info('[{}/{}], [{}/{}], g_step: {}, Spe: {:.1f} sam/sec, l_all: {:.4f}, l_cls: {:.4f}, '
                                 'l_ang: {:.4f}, l_diou: {:.4f}, l_embed: {:.4f}, lr: {:.6}, T: {:.2f}'
                                 .format(str(epoch).zfill(3), self.epochs, str(i + 1).zfill(3), self.train_loader_len,
                                         self.global_step, self.display_interval * cur_batch_size / batch_time,
                                         loss_all, loss_cls, loss_ang, loss_diou, loss_embed, lr, batch_time))
                batch_start = time.time()

            # if self.tensorboard_enable:
            #     self.writer.add_scalar('TRAIN/LOSS/loss_all', loss_all, self.global_step)
            #     self.writer.add_scalar('TRAIN/LOSS/loss_cls', loss_cls, self.global_step)
            #     self.writer.add_scalar('TRAIN/LOSS/loss_ang', loss_ang, self.global_step)
            #     self.writer.add_scalar('TRAIN/LOSS/loss_diou', loss_diou, self.global_step)
            #     self.writer.add_scalar('TRAIN/LOSS/loss_embed', loss_embed, self.global_step)
            #     self.writer.add_scalar('TRAIN/lr', lr, self.global_step)

        # print(f"Finished {epoch=}")
        return {'train_loss': train_loss / self.train_loader_len, 'lr': lr, 'time': time.time() - epoch_start, 'epoch': epoch}

    def _eval_map(self, use_previous_predictions=False): #TODO change this to False
        np.seterr("raise") #TODO remove this
        
        print(f"\n\n\nStaring the evaluation at {datetime.datetime.now()}");t=time.time()
        self.logger.info('Enter evaluating...')
        self.model.eval()
        result_save_path = os.path.join(self.save_dir, 'result')
        if not use_previous_predictions:
            if os.path.exists(result_save_path):
                shutil.rmtree(result_save_path, ignore_errors=True)
            if not os.path.exists(result_save_path):
                os.makedirs(result_save_path)
        print("Time to start up:", time.time()-t);t=time.time()
        
        predict_embeddings, joint_boxes, all_gt_boxes = [], [], []
        qbs_words, qbs_queries, qbs_targets, db_targets, gt_targets = [], [], [], [], []
        overlaps, used_test_word = [], []

        # Compute a mapping from class string to class id...
        self.label_encoder.fit([word for word in self.test_words])
        print("Time for the label encoder:", time.time()-t);t=time.time()

        # Create queries...
        test_unique_words, counts = np.unique(self.test_words, return_counts=True)
        for idx, test_word in enumerate(self.test_words):
            gt_targets.extend(self.label_encoder.transform([test_word]))
            if test_word not in used_test_word and test_word in test_unique_words:
                qbs_words.append(test_word)
                qbs_queries.append(self.words_embeddings[test_word])
                qbs_targets.extend(self.label_encoder.transform([test_word]))
                used_test_word.append(test_word)
        print("Time to create the queries:", time.time()-t)

        for i, (img_file, gt_file) in enumerate(zip(self.test_img_files, self.test_gt_files)):
            self.logger.info('Evaluating {} image: {}'.format(i, img_file))
            print('Evaluating {} image: {}'.format(i, img_file));t=time.time()
            # Get test gt boxes & gt words...
            gt_boxes, gt_words = [], []
            with open(gt_file, mode='r', encoding='utf-8') as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip().rstrip('\n').lstrip('\ufeff').strip().split(',', maxsplit=8)
                gt_boxes.append([int(ver) for ver in line[:8]])
                gt_words.append(str(line[-1]).strip().lower())
            print("Time to load gt:", time.time()-t);t=time.time()

            if not use_previous_predictions:
                # Get img...
                im = Image.open(img_file)
                im = im.convert("RGB")
                im, ratio_w, ratio_h = resize_img(im, long_size=self.config['trainer']['long_size'])
                print("Time to get img and resize it:", time.time()-t);t=time.time()
                
                with torch.no_grad():
                    if str(self.device).__contains__('cuda'):
                        torch.cuda.synchronize(self.device)
                    print("Time to synchronize the device:", time.time()-t);t=time.time()
                    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
                    im = transform(im).unsqueeze(0)
                    im = im.to(self.device)
                    print("Time to transform img:", time.time()-t);t=time.time()
                    (predict_score, predict_geo), predict_embed = self.model(im)
                    print("Time to predict:", time.time()-t);t=time.time()
                    if str(self.device).__contains__('cuda'):
                        torch.cuda.synchronize(self.device)
                    print("Time to synchronize the device again:", time.time()-t);t=time.time()
                # Predicting boxes...
                predict_boxes, _ = get_boxes(score=predict_score.squeeze(0).cpu().numpy(), geo=predict_geo.squeeze(0).cpu().numpy(),
                                             cls_score_thresh=self.cls_score_thresh, bbox_nms_overlap=self.bbox_nms_overlap)
                predict_embed = predict_embed.squeeze(0).cpu().numpy()
                print(f"Found {predict_boxes.shape} boxes and {predict_embed.shape} embeddings")
                print("Time to get the boxes from the feature map:", time.time()-t);t=time.time()
                print(f"The predicted embeddings have shape {predict_embed.shape}")

                if predict_boxes is None or len(predict_boxes) == 0:
                    continue
                self.logger.info('Idx: {0} ===> Predict result [predict_boxes: {1}; gt_boxes: {2}]'.format(i, predict_boxes.shape, len(gt_boxes)))
                valid_prediction_boxes = np.zeros(len(predict_boxes), dtype=np.bool_)
                for i,predict_box in enumerate(predict_boxes):
                    min_x = min(predict_box[0], predict_box[2], predict_box[4], predict_box[6])
                    max_x = max(predict_box[0], predict_box[2], predict_box[4], predict_box[6])
                    min_y = min(predict_box[1], predict_box[3], predict_box[5], predict_box[7])
                    max_y = max(predict_box[1], predict_box[3], predict_box[5], predict_box[7])
                    w, h = max_x - min_x, max_y - min_y
                    differ = h * 0.2 if h < w else w * 0.2
                    min_x, max_x = int((min_x + differ) / 4), int((max_x - differ) / 4)
                    min_y, max_y = int((min_y + differ) / 4), int((max_y - differ) / 4)
                    if min_x >= max_x or min_y >= max_y:
                        continue
                    valid_prediction_boxes[i] = True
                    # print(f"The shape of the box of the predicted embedding is {predict_embed[:, min_y:max_y, min_x:max_x].shape}")
                    if len(predict_embed[:, min_y:max_y, min_x:max_x]) > 0:
                        predict_embeddings.append(np.mean(predict_embed[:, min_y:max_y, min_x:max_x], axis=(1, 2)))
                print("Time to compute embeddings for the predictions", time.time()-t);t=time.time()

                predict_boxes = predict_boxes[valid_prediction_boxes]
                predict_boxes = adjust_ratio(predict_boxes, ratio_w, ratio_h)
                seq = []
                if predict_boxes is not None:
                    seq.extend([','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in predict_boxes])
                seqfile = os.path.join(result_save_path, str(os.path.basename(img_file).split('.')[0]) + '.txt')
                with open(seqfile, 'w') as f:
                    f.writelines(seq)
                print(f"Time to write them in {seqfile}:", time.time()-t);t=time.time()

            else:
                raise Exception("invalid code path")
                seqfile = os.path.join(result_save_path, str(os.path.basename(img_file).split('.')[0]) + '.txt')
                if not os.path.exists(seqfile): continue
                predict_boxes = np.loadtxt(seqfile, dtype=np.int32, delimiter=',')

            joint_boxes.extend(predict_boxes[:, :8])
            all_gt_boxes.extend(gt_boxes)
            gt_boxes = np.array(gt_boxes)
            # Calculate overlap...
            overlap = cal_overlap(predict_boxes, gt_boxes)
            overlaps.append(overlap)
            inds = overlap.argmax(axis=1)
            db_targets.extend(self.label_encoder.transform([gt_words[idx] for idx in inds]))
            print("Time to compute overlap:", time.time()-t);t=time.time()
        
        print("Setting up db on disk...")
        db_path = os.path.join(self.save_dir, "db.npy")
        if not use_previous_predictions:
            db = np.vstack(predict_embeddings) if len(predict_embeddings) != 0 else np.array(predict_embeddings)
            np.save(db_path, db)
        else:
            db = np.load(db_path)
            
        # End evaluate...
        all_overlaps = np.zeros((len(joint_boxes), len(all_gt_boxes)), dtype=np.float64)
        x, y = 0, 0
        for o in overlaps:
            all_overlaps[y:y + o.shape[0], x: x + o.shape[1]] = o
            y += o.shape[0]
            x += o.shape[1]
        db_targets, qbs_targets, qbs_words = np.array(db_targets), np.array(qbs_targets), np.array(qbs_words)
        qbs_queries, joint_boxes = np.array(qbs_queries), np.array(joint_boxes)

        assert (qbs_queries.shape[0] == qbs_targets.shape[0])
        print("db size", db.shape[0])
        # print("num of test images", len(self.test_img_files))
        # print("num of test label files", len(self.test_gt_files))
        print("db targets", db_targets.shape[0])
        assert (db.shape[0] == db_targets.shape[0])

        t=time.time()
        self.logger.info('Calculate mAP...')
        mAP_qbs, mR_qbs = cal_map(qbs_queries, qbs_targets, db, db_targets, gt_targets, joint_boxes,
                                  all_overlaps, self.query_nms_overlap, self.overlap_thresh, qbs_words, num_workers=3)#self.config["trainer"]["num_workers"]) #TODO more workers??
        mAP_qbs, mR_qbs = np.mean(mAP_qbs * 100), np.mean(mR_qbs * 100)
        print("Time to calculate map:", time.time()-t);t = time.time()
        
        # Calculate recall precision f1
        res_dict = cal_recall_precison_f1(gt_path=self.test_gt_path, result_path=result_save_path)
        print("Time to calculate recall precision f1:", time.time()-t)
        return res_dict['recall'], res_dict['precision'], res_dict['hmean'], mAP_qbs, mR_qbs

    def _on_epoch_finish(self, epoch):
        # gc.collect()
        # torch.cuda.empty_cache()
        self.logger.info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['time'], self.epoch_result['lr']))

        if epoch % self.save_interval == 0:
            net_save_path = '{}/WordRetrievalNet_latest.pth'.format(self.checkpoint_dir)
            self._save_checkpoint(self.epoch_result['epoch'], net_save_path)

            save_best = False
            if self.config['trainer']['metrics'] == 'map':  #Use map as optimal model indicator
                recall, precision, hmean, mAP_qbs, mR_qbs = self._eval_map()

                if self.tensorboard_enable:
                    self.writer.add_scalar('EVAL/precision', precision, self.global_step)
                    self.writer.add_scalar('EVAL/recall', recall, self.global_step)
                    self.writer.add_scalar('EVAL/hmean', hmean, self.global_step)
                    self.writer.add_scalar('EVAL/mAP', mAP_qbs, self.global_step)
                    self.writer.add_scalar('EVAL/mR', mR_qbs, self.global_step)
                self.logger.info('test: precision: {:.6f}, recall: {:.6f}, f1: {:.6f}, map: {:.2f}, mr: {:.2f}'.format(precision, recall, hmean, mAP_qbs, mR_qbs))

                if mAP_qbs > self.metrics['map']:
                    save_best = True
                    self.metrics['train_loss'], self.metrics['best_model'] = self.epoch_result['train_loss'], net_save_path
                    self.metrics['precision'], self.metrics['recall'], self.metrics['hmean'] = precision, recall, hmean
                    self.metrics['map'], self.metrics['mr'] = mAP_qbs, mR_qbs
            else:
                if self.epoch_result['train_loss'] < self.metrics['train_loss']:
                    save_best = True
                    self.metrics['train_loss'], self.metrics['best_model'] = self.epoch_result['train_loss'], net_save_path
            if save_best: 
                self.logger.info("Found best model! About to save it")
                self.logger.info(f"{mAP_qbs}>best_MAP={self.metrics['map']}")
            else:
                self.logger.info(f"{mAP_qbs}<=best_MAP={self.metrics['map']}")
            self._save_checkpoint(self.epoch_result['epoch'], net_save_path, save_best)

    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger.info('{}:{}'.format(k, v))
        self.logger.info('Finish train.')

    def _log_memory_usage(self):
        if not self.with_cuda:
            return
        usage = []
        for deviceID, device in self.gpus.items():
            allocated = torch.cuda.memory_allocated(int(deviceID)) / (1024 * 1024)
            cached = torch.cuda.memory_cached(int(deviceID)) / (1024 * 1024)
            usage.append('    CUDA: {0}; Allocated: {1} MB; Cached: {2} MB \n'.format(device, allocated, cached))
        self.logger.debug("Memory Usage: \n{}".format(''.join(usage)))

    def _save_checkpoint(self, epoch, file_name, save_best=False):
        """ Saving checkpoints """
        state_dict = {
            'epoch': epoch,
            'global_step': self.global_step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': self.metrics,
        }
        filename = os.path.join(self.checkpoint_dir, file_name)
        torch.save(state_dict, filename)
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'WordRetrievalNet_best.pth')
            shutil.copy(filename, best_path)
            self.logger.info("Saving current best: {}".format(best_path))
        else:
            self.logger.info("Saving checkpoint: {}".format(filename))

    def _load_checkpoint(self, checkpoint_path, resume):
        """ Resume from saved checkpoints """
        self.logger.info("Loading checkpoint: {}...".format(checkpoint_path))
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
        if resume:
            self.global_step = checkpoint['global_step']
            self.start_epoch = checkpoint['epoch'] #+ 1
            self.logger.info(f"Resuming from {checkpoint['epoch']}")
            self.config['lr_scheduler']['args']['last_epoch'] = self.start_epoch
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'metrics' in checkpoint:
                self.metrics = checkpoint['metrics']
            if self.with_cuda:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
            self.logger.info("Resume from checkpoint {} (epoch {})".format(checkpoint_path, self.start_epoch))
        else:
            self.logger.info("FineTune from checkpoint {}".format(checkpoint_path))

    def _initialize(self, name, module, *args, **kwargs):
        module_name = self.config[name]['type']
        module_args = self.config[name]['args']
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def adjust_learning_rate(self, epoch):
        if epoch in self.lr_step:
            self.lr = self.lr * 0.1
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.lr



def signal_handler(signum, stack):
    raise KeyboardInterrupt

signal.signal(signal.SIGUSR1, signal_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    # parser.add_argument("-c", "--continue", help="continue from latest checkpoint")
    parser.add_argument("-o", "--output")
    parser.add_argument("-r", "--resume-checkpoint")
    parser.add_argument("-f", "--finetune-checkpoint")
    parser.add_argument("-e", "--epochs")
    
    args = parser.parse_args()

    global_cfg = pick_configuration(
        args.dataset,
        trainer_output_dir=args.output,
        trainer_resume_checkpoint=args.resume_checkpoint,
        trainer_finetune_checkpoint=args.finetune_checkpoint,
        epochs=int(args.epochs),
    )
    
    print(global_cfg)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in global_cfg['trainer']['gpus']])
    
    trainer = Trainer(global_cfg)
    
    
    start_at = time.monotonic()
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Received keyboard interruption")
        net_save_path = '{}/WordRetrievalNet_latest.pth'.format(trainer.checkpoint_dir)
        trainer._save_checkpoint(trainer.epoch_result['epoch'], net_save_path)
        print("Saved checkpoint")
    finished_at = time.monotonic()
    
    print(f"The training took {timedelta(seconds=finished_at-start_at)}")
