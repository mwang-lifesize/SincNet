#!/usr/bin/env python3

# run under pytorch environment
# python3 final_test.py /var/www/html/record/testing_data /home/mwang/Development/deep-learning/SincNet/model_raw.pkl.lifesize 

import shutil
import os
import soundfile as sf
import numpy as np
import sys

from predict_cpu import predict_model

#pred_model = predict_model("/home/mwang/Development/deep-learning/SincNet/model_raw.pkl.lifesize")

# handle one dir 
def evaludate_dir(in_dir, pred_model):
  correct_predict = 0
  total_predict = 0
  for u_dir in os.listdir(in_dir):
    speaker_name = str(u_dir)
    print("speaker name:" + speaker_name)
    user_dir = os.path.join(in_dir, u_dir)
    
    if os.path.isdir(user_dir):
      # checking this dir
      this_usr_correct_predict = 0
      this_usr_total_predict = 0
      files = os.listdir(user_dir)
      for file in files:
        file_name = str(file)
        fname = os.path.join(user_dir, file)
        print("\nfname : " + fname)
        if os.path.isfile(fname):
          fn = fname.split('/')
          fn = fn[-1]
          # wave file
          if fn[-4:]=='.wav':
            user,prob = pred_model.predict(fname)
            this_usr_total_predict += 1 
            if user == speaker_name :
              print( " predict ok")
              this_usr_correct_predict += 1
      this_usr_p = float(this_usr_correct_predict)/float(this_usr_total_predict)
      print( "predict for:", speaker_name,  " correct prob is:", this_usr_p)
      correct_predict += this_usr_correct_predict
      total_predict += this_usr_total_predict

  p = float(correct_predict)/float(total_predict)
  print( "predict for all users correct prob is:", this_usr_p)

testing_data=sys.argv[1]
testing_model=sys.argv[2]

print("tesing_data folder:", testing_data)
print("tesing_model :", testing_model)

pred_model = predict_model(testing_model)

# Replicate input folder structure to output folder
#copy_folder(in_folder,out_folder)

evaludate_dir( testing_data, pred_model )

