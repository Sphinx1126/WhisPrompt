# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 19:33:28 2024

@author: 28257
"""

import os
import argparse
import random
import whisper
import pickle

def set_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=r'data/',type=str)
    parser.add_argument('--output_dir', default=r'datasets/',type=str)
    parser.add_argument('--cls_num', default=6,type=int)
    args = parser.parse_args()
    return args


def get_mel(data_dir,cls_num):
    data,labels=[],[]
    for i in range(args.cls_num):
        filenames=os.listdir(data_dir+str(i)+'/')
        data+=filenames
        labels+=[i]*len(filenames)
    
    audio_list=[]
    for file,label in zip(data,labels):
        path=data_dir+str(label)+'/'+file
        
        audio=whisper.load_audio(path)
        audio = whisper.pad_or_trim(audio,length=16000*30)
        mel = whisper.log_mel_spectrogram(audio)
        
        audio_list.append(mel)
    return audio_list,labels

def write_data(data_dir,output_dir,cls_num,mode):
    data,labels=get_mel(args.data_dir+mode+'/',args.cls_num)
    with open(output_dir+mode+'.pkl', 'wb') as fp:
        pickle.dump([data,labels], fp)


if __name__ == '__main__':
    args = set_args()
    
    write_data(args.data_dir,args.output_dir, args.cls_num, 'train')
    write_data(args.data_dir,args.output_dir, args.cls_num, 'valid')
    write_data(args.data_dir,args.output_dir, args.cls_num, 'test')