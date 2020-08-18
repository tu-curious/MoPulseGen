# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 20:46:09 2020

@author: agarwal.270a
"""


import numpy as np
import matplotlib.pyplot as plt
#from scipy import io
#from pathlib import Path
from simulator_for_CC import Simulator
import tensorflow as tf
from tensorflow.keras import layers
import glob
import pandas as pd

class HR2Rpeaks_Simulator(Simulator):
    '''
    Produces peak train from given smooth HR data
    '''
    def __init__(self,input_list,output_list,HR_win_len,w_pk=1,w_l=0.4,P_ID='',path='./',
                 Fs_HR=25):
        '''
        in/output format: list of numpy arrays
        '''
        super(HR2Rpeaks_Simulator,self).__init__(input_list,output_list)
        self.w_pk=w_pk
        self.w_l=w_l
        self.P_ID=P_ID
        self.path=path
        self.Fs_HR=Fs_HR
        self.HR_win_len=HR_win_len
        
        # gen_model is a 2nd order p(RR_prev,RR_next| HR cluster) distribution
        self.gen_model_path=path+"model_weights/{}_HRV_gen_model.mat".format(P_ID)
        
        fname=glob.glob(self.gen_model_path+'*.index')
        if len(fname)==0:
            print('\n Learning HRV Gen Model... \n')
            self.gen_model=self.learn_gen_model(save_flag=True)
            del self.gen_model
        
        #Load and Test
        self.RNN_win_len=1# Reduce RNN win_len to reduce test time wastage
        self.win_step_size=1*self.RNN_win_len
        print('HRV Gen Model Exists. Loading ...')
        self.gen_model=self.load_gen_model(self.gen_model_path,
                            self.Fs_HR)
        print('Done!')
        #self.make_gen_plots(self.ecg_gen_model,int(self.w_pk*self.Fs_ecg))
        return
    
    def load_gen_model(self,path,Fs_out,stateful=True):
        '''
        Load a model from the disk.
        '''
        RNN_win_len=int(self.RNN_win_len)
        if Fs_out==100:
            model=self.create_gen_model(shape_in=[(None,RNN_win_len,2)],
                                    shape_out=[(None,RNN_win_len,1)],stateful=stateful,
                                    batch_size=1)
        elif Fs_out==25:
            model=self.create_gen_model(shape_in=[(None,RNN_win_len,2)],
                                shape_out=[(None,RNN_win_len,1)],stateful=stateful,
                                batch_size=1)
        else:
            raise AssertionError('Fs_out can only be 25 or 100 at this time.')  
            
        model.load_weights(path)
        return model
    
    
    def create_gen_model(self,shape_in,shape_out,
                     model_path='',stateful=False,batch_size=1,
                     optimizer=tf.keras.optimizers.Adam()):
        if stateful:
            print('\n Creating Stateful LSTM in inference Mode. \n')
            inputs = tf.keras.Input(batch_shape=[batch_size]+
                                    list(shape_in[0][1:]))
        else:
            inputs = tf.keras.Input(shape=shape_in[0][1:])
            
        lstm_out=layers.GRU(16,return_sequences=True,
                             stateful=stateful)(inputs)
        out=layers.Conv1D(1,1)(lstm_out)
        
        model= tf.keras.Model(inputs=inputs,outputs=out)
        loss= tf.keras.losses.MeanSquaredError()
        metrics= ['mse']
        model.compile(loss=loss,optimizer=optimizer,metrics=metrics)
        model.summary()
        print(model.optimizer)
        #tf.keras.utils.plot_model(model, "simple_stitcher.png")
        return model
    
    def learn_gen_model(self,save_flag=True,make_plots=True,EPOCHS = 100):
        input_list,output_list=self.input,self.output
        path4model=self.gen_model_path
        #Pre-process data
        model_HR_in,model_RR_in,model_RR_out=[],[],[]
        RNN_win_len,step_size=50,10
        list_HR,list_arr_pks=input_list,output_list
        for j in range(len(list_HR)):
            HR=list_HR[j][list_arr_pks[j].astype(bool)]
            #Get RR_ints in seconds
            RR_ints=Rpeaks2RRint(list_arr_pks[j])
            RR_ints_prev=np.concatenate([RR_ints[0:1],RR_ints[:-1]],axis=0)
            HR=HR[:-1]/60#remove last HR as unusable and convert to BPS
            print(len(RR_ints)-len(HR))
            HR,RR_ints_prev,RR_ints=self.sliding_window_fragmentation(
                                    [HR,RR_ints_prev,RR_ints],RNN_win_len,step_size)
            model_HR_in.append(HR)
            model_RR_in.append(RR_ints_prev)
            model_RR_out.append(RR_ints)
        model_HR_in=np.concatenate(model_HR_in,axis=0)
        model_RR_in=np.concatenate(model_RR_in,axis=0)
        model_RR_out=np.concatenate(model_RR_out,axis=0)
        model_in=np.concatenate([model_HR_in,model_RR_in],axis=-1)
        model_out=model_RR_out
        print(model_in.shape,model_out.shape)

        #partition
        val_perc=0.14
        val_idx=int(val_perc*len(model_in))
        val_data=[model_in[0:val_idx],model_out[0:val_idx]]
        train_data=[model_in[val_idx:],model_out[val_idx:]]
        
        #shuffle AFTER partition as time series based
        perm=np.random.permutation(len(train_data[1]))
        train_data=[train_data[0][perm],train_data[1][perm]]
        perm=np.random.permutation(len(val_data[1]))
        val_data=[val_data[0][perm],val_data[1][perm]]

        optimizer=tf.keras.optimizers.Adam()
            
        model=self.create_gen_model(shape_in=[train_data[0].shape], 
                                   shape_out=[train_data[1].shape],
                                   optimizer=optimizer)
        callbacks=[]
        if save_flag:
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                            filepath=path4model,
                                            save_weights_only=True,
                                            monitor='val_loss',
                                            mode='min',
                                            save_best_only=True)
            callbacks.append(model_checkpoint_callback)
            
        model.fit(train_data[0],train_data[1],batch_size=64,epochs=EPOCHS,
                  validation_data=tuple(val_data),
                  callbacks=callbacks)
        
        #if make_plots:
         #   self.make_stitchGAN_plots(model,w_pk)
        self.model_in=model_in
        self.model_out=model_out
        return model
            
# =============================================================================
#     def sample_RR_old(self,HR,RR_prev):
#         #get cluster
#         HR_up=(self.HR_clusters>HR).astype(int)
#         z=(np.arange(len(self.HR_clusters)-1))[(np.diff(HR_up)).astype(bool)][0]
#         #RR_z=self.RR_distro[z]
#         RR_z_distro=self.RR_distro[z,0];RR_z_vals=self.RR_distro[z,1].reshape(-1)
#         if RR_prev==0: #beginning of sampling. sample uniform randomly
#             RR_next=RR_z_vals[np.random.randint(len(RR_z_vals))]
#         else:
#             idx_Rp=np.arange(len(RR_z_vals))[RR_z_vals==RR_prev]
#             RR_z_Rp=RR_z_distro[idx_Rp,:] #conditional distro given z, RR_p
#             
#             #sample RR_next idx
#             idx_Rn=np.random.choice(len(RR_z_vals),p=RR_z_Rp/np.sum(RR_z_Rp))
#             RR_next=RR_z_vals[idx_Rn]
#         return RR_next
# =============================================================================

# =============================================================================
#     def sample_RR_old(self,HR,RR_prev):
#         list_pdf_RR_joint=[self.RR_distro[j,0] for j in 
#                            range(len(self.RR_distro))]
#         list_pdf_RR_row_sum=[np.sum(arr,axis=0) for arr in list_pdf_RR_joint]
#         #get cluster
#         HR_up=(self.HR_clusters>HR).astype(int)
#         #print(HR_up.shape,(np.diff(HR_up)).astype(bool))
#         z=(np.arange(len(self.HR_clusters)-1))[(np.diff(HR_up)).astype(bool)][0]
#         #get distros
#         RR_z_distro=list_pdf_RR_row_sum[z]
#         RR_z_vals=self.RR_distro[z,1].reshape(-1)
#         #sample
#         idx_Rn=np.random.choice(len(RR_z_vals),p=RR_z_distro) #sample RR_next idx
#         RR_next=RR_z_vals[idx_Rn]
#         return RR_next
# =============================================================================
    
    def sample_RR(self,HR,RR_prev):
        RR_next=self.gen_model.predict(np.array([HR,RR_prev]).reshape(1,1,2))
        return RR_next[0,0,0]
    
    def __call__(self,HR_curve,Fs_out=100):
        factr=(Fs_out/self.Fs_HR)
        arr_pk=np.zeros(int(factr*len(HR_curve)))
        #TODO: bunch of changes here
        #gauss=norm(loc = 0., scale = 1.5).pdf(np.arange(-3,3+1))
        #plt.figure();plt.plot(gauss)
        i=int(self.Fs_HR*self.w_l)
        avg_HR=np.mean(HR_curve[0:i])
        RR_next=60/avg_HR
        w_r=int(((self.w_pk-self.w_l)*self.Fs_HR))-1
        while i < (len(HR_curve)-w_r-1):
            idx=int(factr*i)
            arr_pk[idx]=1
            #get next RR_interval
            #avg_HR=np.mean(HR_curve1[i-w_l:i+w_r+1])
            RR_prev=RR_next*1
            look_ahead=int(self.Fs_HR*(60/avg_HR))#based on last HR
            if i > self.HR_win_len:
                avg_HR=np.mean(HR_curve[i+look_ahead-self.HR_win_len:i+look_ahead])
            else:
                avg_HR=np.mean(HR_curve[0:i+look_ahead]) #look ahead HR
            RR_next=self.sample_RR(avg_HR/60,RR_prev)
            i+=int(RR_next*self.Fs_HR)
        return arr_pk

#%% Sample Client
if __name__=='__main__':
    #import pandas as pd
    #import glob
    from data.sim_for_model_4 import HR_func_generator
    from HR2ecg_ppg_Simulator import filtr_HR,arr_t
    
    def Rpeak2HR(test_pks,win_len=8*100,step=1,Fs_pks=100):
        HR_curve=[np.sum(test_pks[step*i:step*i+win_len])/(win_len/Fs_pks) for i in 
                  range(int((len(test_pks)-win_len+1)/step))]
        HR_curve=filtr_HR(np.array(HR_curve)*60)
        return HR_curve
    
    def Rpeaks2RRint(arr_pks, Fs_pks=100):
        r_pk_locs_origin=np.arange(len(arr_pks))[arr_pks.astype(bool)]
        RR_ints=np.diff(r_pk_locs_origin/Fs_pks).reshape((-1,1))
        return RR_ints
    
    def get_train_data(path,val_files=[],test_files=[],
                       win_len=8*100,step=1,Fs_pks=100):
        '''
        Use all files in the folder 'path' except the val_files and test_files
        '''
        def get_clean_ppg_and_ecg(files):
            list_arr_pks=[]
            for i in range(len(files)):
                df=pd.read_csv(files[i],header=None)
                arr=df.values
                if 'clean' in files[i]:
                    list_arr_pks+=[arr[:,45:49].reshape(-1)]    
            return list_arr_pks
        files=glob.glob(path+'*.csv')
        #files=[fil for fil in files if 'WZ' in fil] #get wenxiao's data
        #separate val and test files
        s3=set(files);s4=set(val_files+test_files)
        files_2=list(s3.difference(s4))
        #files_2=[fil for fil in files if not((val_names[0] in fil))]
        list_arr_pks=get_clean_ppg_and_ecg(files_2)
        
        list_HR=[Rpeak2HR(arr_pks,win_len,step,Fs_pks) for arr_pks in list_arr_pks]
        
        #select HR only at peak locations for training
        n_HR=[int((len(arr_pks)-win_len+1)/step) for arr_pks in list_arr_pks]
        list_arr_pks=[list_arr_pks[j][-n_HR[j]:] for j in  range(len(list_arr_pks))]

        return list_HR,list_arr_pks

    def get_test_data(file_path,win_len,step,Fs_pks):
        df=pd.read_csv(file_path,header=None)
        arr=df.values
        test_out_for_check=arr[:,45:49].reshape(-1)
        test_in=Rpeak2HR(test_out_for_check,win_len,step,Fs_pks)
        n_HR=int((len(test_out_for_check)-win_len+1)/step)
        return test_in.reshape(-1).astype('float32'),test_out_for_check[-n_HR:].astype('float32')

    #Get Train Data for simulator
    plt.close('all')
    path_prefix= 'E:/Box Sync/' #'C:/Users/agarwal.270/Box/' #
    path=(path_prefix+'AU19/Research/PPG_ECG_proj/data/Wen_data_28_Sep/'
          'clean_lrsynced\\')
    val_files=[path+'2019092801_3154_clean.csv']
    test_files=[path+'2019092820_5701_clean.csv']
    win_len=8*100;step=1;Fs_pks=100
    input_list,output_list=[],[]
    input_list,output_list=get_train_data(path,val_files,test_files,win_len,
                                           step,Fs_pks)
    
    # =============================================================================
    # check1=output_list[0]
    # check1[:,1:5]=(detrend(check1[:,1:5].reshape(-1),0,'constant')).reshape((-1,4))
    # plt.figure();plt.plot(check1[:,0:1])
    # plt.figure();plt.plot(check1[:,1:5].reshape(-1))
    # =============================================================================
    
    #Generate HR Data
    #Fs=100
    #t=arr_t[np.random.randint(len(arr_t))] # sample seq. length in s.
    #t1=np.linspace(0,t,num=int(t*Fs),endpoint=False)
    #HR_curve_f,D_HR=HR_func_generator(t1)
    
    

    #plt.figure();plt.plot(HR_curve)
    #synth_ecg_out,test_in_ecg,synth_ppg_out,test_in_ppg=sim_pks2sigs(test_in)
    
    #define constants
    path_prefix=  'E:/Box Sync/' #'C:/Users/agarwal.270/Box/'
    path=path_prefix+'SP20/Research/PPG_ECG_proj/simulator_CC/data/'
    P_ID='W'
    Fs=100 #Hz
    #Train
    sim_HR2pks=HR2Rpeaks_Simulator(input_list,output_list,HR_win_len=win_len,
                                   path=path,
                                   P_ID=P_ID,Fs_HR=Fs)

    #Test Simulator
    #Use simulator to produce synthetic output given input
    
    test_in,test_out_for_check=get_test_data(test_files[0],win_len=8*100,step=1,Fs_pks=100)
    Fs_ppg=25;Fs_ecg=100
    arr_pk=sim_HR2pks(test_in)
    
    #Plot some stuff
    plt.figure()  
    plt.plot(test_out_for_check[19:])
    plt.plot(arr_pk)
    plt.legend(['True','Synthetic'])
    
# =============================================================================
#     #generate HR
#     len_in_s=20.48 #s
#     len_out=4
#     len_in=Fs*len_in_s
#     arr_t=np.arange(250,900,len_in_s) #change time duration when longer noise exists
#     t=arr_t[np.random.randint(len(arr_t))] # sample seq. length in s.
#     t1=np.linspace(0,t,num=int(t*Fs),endpoint=False)
#     HR_curve_f,D_HR=HR_func_generator(t1)
#     
#     #Test Simulator
#     sim_HR2pks=HR2Rpeaks_Simulator([],[],path=path,P_ID=P_ID,Fs_HR=Fs)
#     arr_pk=sim_HR2pks(HR_curve_f)
#     
#     #Check if upsampling works
#     arr_pk_upsampled=sim_HR2pks(HR_curve_f,Fs_out=100)
#     check=arr_pk_upsampled.reshape(-1,4)
#     plt.figure();plt.plot(arr_pk);plt.plot(arr_pk_upsampled[::4])
#     #ppg1,HR1=gen_ppg_from_HR(t1,HR_curve_f,D_HR,peak_id,make_plots=make_plots)
# =============================================================================
