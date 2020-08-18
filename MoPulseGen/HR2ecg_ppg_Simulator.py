# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 22:28:40 2020

@author: agarwal.270a
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import detrend
from simulator_for_CC import Simulator
from HR2Rpeaks_v2 import HR2Rpeaks_Simulator
from Rpeaks2EcgPpg_gen_model_v23 import Rpeaks2EcgPpg_Simulator
import pickle
#del Rpeaks2EcgPpg_Simulator
#del HR2EcgPpg_Simulator

#%%

class HR2EcgPpg_Simulator(Simulator):
    '''
    find peak_train. from there on, more or less ppg ecg
    '''
    def __init__(self,input_list,latent_list,output_list,w_pk=1,w_l=0.4,P_ID=''
                 ,path='./',ppg_id='ppg_green',Fs_HR=25,Fs_pks=100,
                 latent_size=2,ecg_by_ppg_pk=1.):
        
        super(HR2EcgPpg_Simulator,self).__init__(input_list,output_list)
        self.Fs_HR=Fs_HR
        self.Fs_pks=Fs_pks
        self.w_pk=w_pk
        self.w_l=w_l
        self.P_ID=P_ID
        self.path=path
        self.ppg_id=ppg_id
        self.latent_list=latent_list
        self.latent_size=latent_size
        #self.latent_list=self.find_latent_list(output_list[::4])
        self.sim_HR2pks=HR2Rpeaks_Simulator(input_list,self.latent_list,
                                            HR_win_len=8*100,
                                            path=path,P_ID=P_ID,w_pk=w_pk,
                                            Fs_HR=self.Fs_HR,w_l=w_l)
        #Create Simulator using train data
        self.sim_pks2sigs=Rpeaks2EcgPpg_Simulator(self.latent_list,output_list,
                                                  P_ID=P_ID,path=path,
                                                  latent_size=latent_size,
                                                  w_pk=w_pk,w_l=w_l,
                                                  ecg_by_ppg_pk=ecg_by_ppg_pk,
                                                  ppg_id=ppg_id)
        
    def find_latent_list(self,ecg_list):
        '''
        Finds R-peaks from ecg data
        '''
        pass
        
    def __call__(self,HR_curve):
        arr_pk=self.sim_HR2pks(HR_curve,Fs_out=self.Fs_pks)
        arr_ecg,arr_pks_ecg,clip_ecg,arr_ppg,arr_pks_ppg,clip_ppg=self.sim_pks2sigs(arr_pk)
        return arr_ecg,arr_pks_ecg,clip_ecg,arr_ppg,arr_pks_ppg,clip_ppg
    
#%% Sample Client

import glob
import pandas as pd
from data.sim_for_model_4 import HR_func_generator

#define constants
path_prefix=  'E:/Box Sync/' #'C:/Users/agarwal.270/Box/'
path=path_prefix+'SP20/Research/PPG_ECG_proj/simulator_CC/data/'
P_ID='W'
Fs=25 #Hz
len_in_s=20.48 #s
len_out=4
len_in=Fs*len_in_s
arr_t=np.arange(250,900,len_in_s) #change time duration when longer noise exists
max_ecg_val=2**16-1
#Generate ecg and ppg training data

def get_train_data(path,val_files=[],test_files=[]):
    '''
    Use all files in the folder 'path' except the val_files and test_files
    '''
    def get_clean_ppg_and_ecg(files):
        '''
        
        '''
        list_clean_ppg=[];list_arr_pks=[]
        for i in range(len(files)):
            df=pd.read_csv(files[i],header=None)
            arr=df.values
            if 'clean' in files[i]:
                arr[:,41:45]=(detrend(arr[:,41:45].reshape(-1),0,'constant')
                                ).reshape((-1,4))
                list_clean_ppg+=[np.concatenate([arr[:,29:30],arr[:,41:45]],
                                    axis=-1),arr[:,30:31],arr[:,39:40],
                                arr[:,40:41]]
                list_arr_pks+=4*[arr[:,45:49].reshape(-1)]    
        return list_arr_pks,list_clean_ppg
    files=glob.glob(path+'*.csv')
    #files=[fil for fil in files if 'WZ' in fil] #get wenxiao's data
    #separate val and test files
    s3=set(files);s4=set(val_files+test_files)
    files_2=list(s3.difference(s4))
    #files_2=[fil for fil in files if not((val_names[0] in fil))]
    list_arr_pks,list_clean_ppg=get_clean_ppg_and_ecg(files_2)
    return list_arr_pks,list_clean_ppg

def get_test_data(file_path):
    df=pd.read_csv(file_path,header=None)
    arr=df.values
    mean=np.mean(arr[:,41:45].reshape(-1))
    #arr[:,41:45]-=np.ceil(max_ecg_val/2)
    #arr[:,41:45]-=mean
    #arr[:,41:45]/=max_ecg_val
    #arr[:,41:45]-=np.mean(arr[:,41:45].reshape(-1))

    #arr[:,41:45]=(arr[:,41:45]/np.mean(arr[:,41:45].reshape(-1)))-1

    test_out_for_check=[np.concatenate([arr[:,29:30],arr[:,41:45]],axis=-1),
                        arr[:,30:31],arr[:,39:40],arr[:,40:41]]
    test_in=arr[:,45:49].reshape(-1)
    return test_in,test_out_for_check,mean

import scipy.signal as sig
def filtr_HR(X0,Fs=100,filt=True):
    nyq=Fs/2
    if len(X0.shape)==1:
        X0=X0.reshape(-1,1)
    X1 = np.copy(X0)#sig.detrend(X0,type='constant',axis=0); # Subtract mean
    if filt:
        # filter design used from Ju's code with slight changes for python syntax
        b = sig.firls(219,np.array([0,0.5,1,nyq]),np.array([1,1,0,0]),np.array([1,1]),nyq=nyq);
        X=np.zeros(X1.shape)
        for i in range(X1.shape[1]):
            #X[:,i] = sig.convolve(X1[:,i],b,mode='same'); # filtering using convolution, mode='same' returns the centered signal without any delay
            X[:,i] = sig.filtfilt(b,[1],X1[:,i])
    else:
        X=X1
    #X=sig.detrend(X,type='constant',axis=0); # subtracted mean again to center around x=0 just in case things changed during filtering
    return X
#%%
if __name__=='__main__':
    #Get Train Data for simulator
    plt.close('all')
    path_prefix= 'E:/Box Sync/' #'C:/Users/agarwal.270/Box/' #
    path=(path_prefix+'AU19/Research/PPG_ECG_proj/data/Wen_data_28_Sep/'
          'clean_lrsynced\\')
    val_files=[path+'2019092801_3154_clean.csv']
    test_files=[path+'2019092820_5701_clean.csv']
    latent_list,output_list=get_train_data(path,val_files,test_files)
    
    # =============================================================================
    # check1=output_list[0]
    # check1[:,1:5]=(detrend(check1[:,1:5].reshape(-1),0,'constant')).reshape((-1,4))
    # plt.figure();plt.plot(check1[:,0:1])
    # plt.figure();plt.plot(check1[:,1:5].reshape(-1))
    # =============================================================================
    
    #Generate HR Data
    t=arr_t[np.random.randint(len(arr_t))] # sample seq. length in s.
    t1=np.linspace(0,t,num=int(t*Fs),endpoint=False)
    HR_curve_f,D_HR=HR_func_generator(t1)
    
    #Use simulator to produce synthetic output given input
    
    test_pks,test_out_for_check,mean=get_test_data(test_files[0])
    win_len=8*100;step=1
    HR_curve=[np.sum(test_pks[step*i:step*i+win_len])/(win_len/100) for i in 
              range(int((len(test_pks)-win_len+1)/step))]
    #HR_curve=filtr_HR(np.array(HR_curve[::4])*60)
    HR_curve=filtr_HR(np.array(HR_curve)*60)

    #plt.figure();plt.plot(HR_curve)
    #synth_ecg_out,test_in_ecg,synth_ppg_out,test_in_ppg=sim_pks2sigs(test_in)
    
    
    #Test Simulator
    path='E:/Box Sync/SP20/Research/PPG_ECG_proj/simulator_CC/data/'
    sim_HR2ecgppg=HR2EcgPpg_Simulator([],latent_list,output_list,path=path,
                                      P_ID='W',Fs_HR=100)
    synth_ecg_out,test_in_ecg,clip_ecg,synth_ppg_out,test_in_ppg,clip_ppg=sim_HR2ecgppg(HR_curve)
    synth_ecg_out*=max_ecg_val #rescale
    synth_ecg_out+=mean #add back mean
    #synth_ecg_out,test_in_ecg,synth_ppg_out,test_in_ppg=sim_HR2ecgppg(HR_curve_f)
    
#%%
    
    #Visualize when using GT data
    time_vec_100=np.arange(len(synth_ecg_out))/100
    time_vec_25=time_vec_100[::4]
    
    fig1=plt.figure()
    start,end=clip_ppg
    ax=plt.subplot(411)
    plt.plot(time_vec_25,HR_curve[start:end]);plt.grid(True)
    plt.grid(True);plt.title('HR')
    plt.xlabel('Time (s)');plt.ylabel('HR (BPM)')
    
    #fig2=plt.figure()
    start,end=clip_ecg
    plt.subplot(412,sharex=ax)
    #plt.plot(time_vec_100,test_pks[win_len-4:])
    plt.plot(time_vec_100,test_pks[start:end])
    plt.plot(time_vec_100,test_in_ecg)
    plt.legend(['True','Synthetic'])
    #plt.grid(True)
    plt.title('Rpeak-Train')
    #plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    start,end=clip_ppg
    plt.subplot(413,sharex=ax)
    #plt.plot(time_vec_25,(test_out_for_check[0][int(win_len/4)-1:,0:1])[:len(time_vec_25)])
    plt.plot(time_vec_25,test_out_for_check[0][start:end,0:1])
    plt.plot(time_vec_25,synth_ppg_out)
    #plt.plot(time_vec_25,test_in_ppg)
    plt.legend(['True','Synthetic'])
    #plt.legend(['True','Synthetic','R-peaks'])
    #plt.grid(True)
    plt.title('PPG')
    #plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    start,end=clip_ecg
    plt.subplot(414,sharex=ax)
    #plt.plot(time_vec_100,(test_out_for_check[0][int(win_len/4)-1:,1:5].reshape(-1))[:len(time_vec_100)])
    plt.plot(time_vec_100,(test_out_for_check[0][:,1:5].reshape(-1))[start:end])
    plt.plot(time_vec_100,synth_ecg_out)
    plt.legend(['True','Synthetic'])
    #plt.grid(True)
    plt.title('ECG')
    plt.xlabel('Time (s)');plt.ylabel('Magnitude')
    plt.grid(True)
    
    #with open('./figures/fig_HREP_highHR.pickle', 'wb') as file:
     #   pickle.dump([fig1],file)
    # =============================================================================
    # plt.figure()
    # ax=plt.subplot(211)
    # plt.plot(time_vec_25,test_out_for_check[0][int(win_len/4)-1:,0:1],'g')
    # #plt.plot(time_vec_25,test_in_ppg)
    # #plt.legend(['True','Synthetic'])
    # #plt.legend(['True','Synthetic','R-peaks'])
    # plt.grid(True)
    # plt.title('Green PPG Signal from Motion_Sense_2')
    # #plt.xlabel('Time (s)')
    # plt.ylabel('Magnitude')
    # 
    # plt.subplot(212,sharex=ax)
    # plt.plot(time_vec_100,test_out_for_check[0][int(win_len/4)-1:,1:5].reshape(-1))
    # plt.grid(True)
    # plt.title('ECG Signal from Autosense')
    # plt.xlabel('Time (s)');plt.ylabel('Magnitude')
    # =============================================================================
#%%
    #Visualize when using sim data only
    time_vec_100=np.arange(len(synth_ecg_out))/100
    time_vec_25=time_vec_100[::4]
    
    fig1=plt.figure()
    start,end=clip_ppg
    ax=plt.subplot(411)
    plt.plot(time_vec_25,HR_curve_f[start:end]);plt.grid(True)
    plt.grid(True);plt.title('HR')
    plt.xlabel('Time (s)');plt.ylabel('HR (BPM)')
    
    #fig2=plt.figure()
    start,end=clip_ecg
    plt.subplot(412,sharex=ax)
    #plt.plot(time_vec_100,test_pks[win_len-4:])
    plt.plot(time_vec_100,test_in_ecg)
    #plt.grid(True)
    plt.title('Synthetic Rpeak-Train')
    #plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    start,end=clip_ppg
    plt.subplot(413,sharex=ax)
    #plt.plot(time_vec_25,(test_out_for_check[0][int(win_len/4)-1:,0:1])[:len(time_vec_25)])
    plt.plot(time_vec_25,synth_ppg_out)
    #plt.plot(time_vec_25,test_in_ppg)
    #plt.legend(['True','Synthetic','R-peaks'])
    #plt.grid(True)
    plt.title('Synthetic PPG')
    #plt.xlabel('Time (s)')
    plt.ylabel('Magnitude')
    plt.grid(True)
    
    start,end=clip_ecg
    plt.subplot(414,sharex=ax)
    #plt.plot(time_vec_100,(test_out_for_check[0][int(win_len/4)-1:,1:5].reshape(-1))[:len(time_vec_100)])
    plt.plot(time_vec_100,synth_ecg_out)
    #plt.grid(True)
    plt.title('Synthetic ECG')
    plt.xlabel('Time (s)');plt.ylabel('Magnitude')
    plt.grid(True)