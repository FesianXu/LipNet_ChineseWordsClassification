# !/usr/bin/env python
# -*- coding:utf-8 -*-

__author__ = 'FesianXu'
__date__ = '2018/10/17'
__version__ = ''

import numpy as np

class GlobalConfig(object):
  '''
  In this global config file, we set the dataset relevant paths and pre-load the index files that we need in the dataloader.
  as for the hyper-parameters setting, we don't specify them here.
  '''
  __instance = None  # to implement singleton
  __call_cls_count = 0  # the counter of calling for new instances
  __is_init_finished = False
  __is_used_server = True  # use server
  def __init__(self):
    GlobalConfig.__call_cls_count += 1
    GlobalConfig.__is_init_finished = False
    '''
    Beginning the GLOBAL CONFIG
    '''

    self.cls_number = 313
    self.user_name = 'fesian'
    self.project_root_path = '/home/{}/contest_workspace/LipNet_ChineseWordsClassification/'.format(self.user_name)

    self.train_index = np.load(self.project_root_path+'dataset/ctc_labels/index_files/train_index.npy')
    self.eval_index = np.load(self.project_root_path+'dataset/ctc_labels/index_files/eval_index.npy')
    self.test_index = np.load(self.project_root_path+'dataset/ctc_labels/index_files/test_index.npy')
    self.whole_train_index = np.load(self.project_root_path+'dataset/ctc_labels/index_files/whole_train_index.npy')
    
    self.img_root_path = self.project_root_path+'dataset/'
    self.train_img_datapath = self.img_root_path+'center_mouth_rgb/train/'
    self.test_img_datapath = self.img_root_path+'center_mouth_rgb/test/'

    self.word2idx_codebook = np.load(self.project_root_path+'dataset/ctc_labels/word_codebook.npy', allow_pickle=True).item()
    self.idx2word_codebook =  {v: k for k, v in self.word2idx_codebook.items()}

    # lipnet setting
    self.nchannels = 3
    self.seq_max_lens = 16
    self.pic_height = 60
    self.pic_width = 120
    self.lipnet_log_path = self.project_root_path+'logs/lipnet/'
    
    # ctc loss setting
    self.char_cls_number = 431
    self.word_cls_number = 313
    self.cls_blank_number = self.char_cls_number+1
    self.char2idx_codebook = np.load(self.project_root_path+'dataset/ctc_labels/char_codebook.npy', allow_pickle=True).item()
    self.idx2char_codebook =  {v: k for k, v in self.char2idx_codebook.items()}
    self.samples_char_lengths = np.load(self.project_root_path+'dataset/ctc_labels/train_eval_char_length.npy', allow_pickle=True)
    # ctc token passing decoder
    self.ctc_char_classes = np.load(self.project_root_path+'dataset/ctc_labels/ctc_char_classes.npy', allow_pickle=True).item()
    self.ctc_corpos_path = self.project_root_path+'dataset/ctc_labels/dictionary.txt'

    '''
    Ending the GLOBAL CONFIG
    '''
    GlobalConfig.__is_init_finished = True

  def get_model_path(self, model_root_path, comment, exp_num):
    return model_root_path % (comment, exp_num)

  def get_log_path(self, root_path, id):
    return root_path % (id)

  def get_exp_data_path(self, root_path, comment):
    return root_path % (comment)

  '''
  ABOVE ARE ALL CONFIG
  '''

  def __setattr__(self, key, value):
    if GlobalConfig.__is_init_finished is True:
      raise AttributeError('{}.{} is READ ONLY'.format(type(self).__name__, key))
    else:
      self.__dict__[key] = value

  def __new__(cls, *args, **kwargs):
    if GlobalConfig.__instance is None:
      GlobalConfig.__instance = object.__new__(cls)
    return GlobalConfig.__instance

  def callCounter(self):
    print('The Global Config class has been called for %d times' % self.__call_cls_count)
