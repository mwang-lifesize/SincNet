# speaker_id.py
# Mirco Ravanelli 
# Mila - University of Montreal 

# July 2018

# Description: 
# This code performs a speaker_id experiments with SincNet.
 
# How to run it:
# python predict.py --cfg=cfg/SincNet_TIMIT.cfg sound_file

import os
#import scipy.io.wavfile
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from torch.autograd import Variable

import sys
import numpy as np
from dnn_models_cpu import MLP,flip
from dnn_models_cpu import SincNet as CNN 
from data_io import ReadList,read_conf,str_to_bool

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
print()
# To move tensors to the respective device
#torch.rand(10).to(device)
# Or create a tensor directly on the device
#torch.rand(10, device=device)

user_mapping = np.load("/var/www/html/record/user_mapping.npy").tolist()

class predict_model:

  def __init__(self):
    self.device = 'cpu' # torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Reading cfg file
    options=read_conf()

    #[data]
    tr_lst=options.tr_lst
    te_lst=options.te_lst
    pt_file=options.pt_file
    class_dict_file=options.lab_dict
    data_folder=options.data_folder+'/'
    output_folder=options.output_folder

    #[windowing]
    fs=int(options.fs)
    cw_len=int(options.cw_len)
    cw_shift=int(options.cw_shift)

    #[cnn]
    cnn_N_filt=list(map(int, options.cnn_N_filt.split(',')))
    cnn_len_filt=list(map(int, options.cnn_len_filt.split(',')))
    cnn_max_pool_len=list(map(int, options.cnn_max_pool_len.split(',')))
    cnn_use_laynorm_inp=str_to_bool(options.cnn_use_laynorm_inp)
    cnn_use_batchnorm_inp=str_to_bool(options.cnn_use_batchnorm_inp)
    cnn_use_laynorm=list(map(str_to_bool, options.cnn_use_laynorm.split(',')))
    cnn_use_batchnorm=list(map(str_to_bool, options.cnn_use_batchnorm.split(',')))
    cnn_act=list(map(str, options.cnn_act.split(',')))
    cnn_drop=list(map(float, options.cnn_drop.split(',')))


    #[dnn]
    fc_lay=list(map(int, options.fc_lay.split(',')))
    fc_drop=list(map(float, options.fc_drop.split(',')))
    fc_use_laynorm_inp=str_to_bool(options.fc_use_laynorm_inp)
    fc_use_batchnorm_inp=str_to_bool(options.fc_use_batchnorm_inp)
    fc_use_batchnorm=list(map(str_to_bool, options.fc_use_batchnorm.split(',')))
    fc_use_laynorm=list(map(str_to_bool, options.fc_use_laynorm.split(',')))
    fc_act=list(map(str, options.fc_act.split(',')))

    #[class]
    self.class_lay=list(map(int, options.class_lay.split(',')))
    class_drop=list(map(float, options.class_drop.split(',')))
    class_use_laynorm_inp=str_to_bool(options.class_use_laynorm_inp)
    class_use_batchnorm_inp=str_to_bool(options.class_use_batchnorm_inp)
    class_use_batchnorm=list(map(str_to_bool, options.class_use_batchnorm.split(',')))
    class_use_laynorm=list(map(str_to_bool, options.class_use_laynorm.split(',')))
    class_act=list(map(str, options.class_act.split(',')))


    #[optimization]
    lr=float(options.lr)
    batch_size=int(options.batch_size)
    N_epochs=int(options.N_epochs)
    N_batches=int(options.N_batches)
    N_eval_epoch=int(options.N_eval_epoch)
    seed=int(options.seed)


    # training list
    wav_lst_tr=ReadList(tr_lst)
    snt_tr=len(wav_lst_tr)

    # test list
    wav_lst_te=ReadList(te_lst)
    snt_te=len(wav_lst_te)

    # setting seed
    torch.manual_seed(seed)
    np.random.seed(seed)

      
    # Converting context and shift in samples
    self.wlen=int(fs*cw_len/1000.00)
    self.wshift=int(fs*cw_shift/1000.00)

    # Batch_dev
    self.Batch_dev=128

    # Feature extractor CNN
    CNN_arch = {'input_dim': self.wlen,
              'fs': fs,
              'cnn_N_filt': cnn_N_filt,
              'cnn_len_filt': cnn_len_filt,
              'cnn_max_pool_len':cnn_max_pool_len,
              'cnn_use_laynorm_inp': cnn_use_laynorm_inp,
              'cnn_use_batchnorm_inp': cnn_use_batchnorm_inp,
              'cnn_use_laynorm':cnn_use_laynorm,
              'cnn_use_batchnorm':cnn_use_batchnorm,
              'cnn_act': cnn_act,
              'cnn_drop':cnn_drop,          
              }

    self.CNN_net=CNN(CNN_arch)
    self.CNN_net.to(self.device)

    # Loading label dictionary
    lab_dict=np.load(class_dict_file).item()

    DNN1_arch = {'input_dim': self.CNN_net.out_dim,
              'fc_lay': fc_lay,
              'fc_drop': fc_drop, 
              'fc_use_batchnorm': fc_use_batchnorm,
              'fc_use_laynorm': fc_use_laynorm,
              'fc_use_laynorm_inp': fc_use_laynorm_inp,
              'fc_use_batchnorm_inp':fc_use_batchnorm_inp,
              'fc_act': fc_act,
              }

    self.DNN1_net=MLP(DNN1_arch)
    #self.DNN1_net.cuda()
    self.DNN1_net.to(self.device)


    DNN2_arch = {'input_dim':fc_lay[-1] ,
              'fc_lay': self.class_lay,
              'fc_drop': class_drop, 
              'fc_use_batchnorm': class_use_batchnorm,
              'fc_use_laynorm': class_use_laynorm,
              'fc_use_laynorm_inp': class_use_laynorm_inp,
              'fc_use_batchnorm_inp':class_use_batchnorm_inp,
              'fc_act': class_act,
              }


    self.DNN2_net=MLP(DNN2_arch)
    #self.DNN2_net.cuda()
    self.DNN2_net.to(self.device)


    if pt_file!='none':
       print("loading pre trained file", pt_file)
       checkpoint_load = torch.load(pt_file)
       self.CNN_net.load_state_dict(checkpoint_load['CNN_model_par'])
       self.DNN1_net.load_state_dict(checkpoint_load['DNN1_model_par'])
       self.DNN2_net.load_state_dict(checkpoint_load['DNN2_model_par'])


    self.CNN_net.eval()
    self.DNN1_net.eval()
    self.DNN2_net.eval()
    test_flag=1 
    loss_sum=0
    err_sum=0
    err_sum_snt=0

  def predict(self, test_file):
     print("starting test :")
       
     #[fs,signal]=scipy.io.wavfile.read(data_folder+wav_lst_te[i])
     #signal=signal.astype(float)/32768

     [signal, fs] = sf.read( test_file )

     #signal=torch.from_numpy(signal).float().cuda().contiguous()
     signal=torch.from_numpy(signal).float().to(self.device).contiguous()

     # split signals into chunks
     beg_samp=0
     end_samp=self.wlen

     # if reminder is 0, total frame is N_fr, is not will not N_fr+1
     N_fr=int((signal.shape[0]-self.wlen)/(self.wshift))
     remainder = (signal.shape[0]-self.wlen) % (self.wshift)
     total_fr = N_fr
     if remainder > 0:
      total_fr = N_fr + 1
     

     sig_arr=np.zeros([self.Batch_dev,self.wlen])
     #pout=Variable(torch.zeros(N_fr+1,self.class_lay[-1]).float().cuda().contiguous())
     #pout=Variable(torch.zeros(N_fr+1,self.class_lay[-1]).float().to(self.device).contiguous())
     pout=Variable(torch.zeros(total_fr,self.class_lay[-1]).float().to(self.device).contiguous())
     count_fr=0
     count_fr_tot=0
     while end_samp<signal.shape[0]:
         #print("validation end_samp:",  end_samp, ", signal.shape is:", signal.shape[0])
         sig_arr[count_fr,:]=signal[beg_samp:end_samp].cpu()
         beg_samp=beg_samp+self.wshift
         end_samp=beg_samp+self.wlen
         count_fr=count_fr+1
         count_fr_tot=count_fr_tot+1
         if count_fr==self.Batch_dev:
             #inp=Variable(torch.from_numpy(sig_arr).float().cuda().contiguous())
             inp=Variable(torch.from_numpy(sig_arr).float().to(self.device).contiguous())
             #import pdb
             #pdb.set_trace()
             pout[count_fr_tot-self.Batch_dev:count_fr_tot,:]=self.DNN2_net(self.DNN1_net(self.CNN_net(inp)))
             count_fr=0
             sig_arr=np.zeros([self.Batch_dev,self.wlen])

     if count_fr>0:
      #inp=Variable(torch.from_numpy(sig_arr[0:count_fr]).float().cuda().contiguous())
      inp=Variable(torch.from_numpy(sig_arr[0:count_fr]).float().to(self.device).contiguous())
      pout[count_fr_tot-count_fr:count_fr_tot,:]=self.DNN2_net(self.DNN1_net(self.CNN_net(inp)))

     print(pout)

     pred=torch.max(pout,dim=1)[1]

     [val,best_class]=torch.max(torch.sum(pout,dim=0),0)

     print( best_class.item() )
     print( val )

     pout_sm = torch.exp(pout)
     sum_of_pout_for_all_chunks = torch.sum(pout_sm, dim=0)/torch.sum(pout_sm)
     best_user = 0
     best_prob = sum_of_pout_for_all_chunks[best_class]
     best_user = user_mapping[best_class.item()]
     print( user_mapping )
     #loss_sum=loss_sum+loss.detach()
     #err_sum=err_sum+err.detach()
      
     print('pred : {}, best_class: {}, best_user: {}, best_probability: {} \n'.format(pred, best_class, best_user, best_prob))



test_file = sys.argv[1]
pred_model = predict_model()
pred_model.predict(test_file)


