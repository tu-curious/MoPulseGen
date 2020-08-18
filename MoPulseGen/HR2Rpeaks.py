# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 20:46:09 2020

@author: agarwal.270a
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from pathlib import Path
from simulator_for_CC import Simulator



class HR2Rpeaks_Simulator(Simulator):
    '''
    Produces peak train from given smooth HR data
    '''
    def __init__(self,input_list,output_list,w_pk=1,w_l=0.4,P_ID='',path='./',
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
        
        # gen_model is a 2nd order p(RR_prev,RR_next| HR cluster) distribution
        self.distro_path=path+"{}_HRV_O2_distro.mat".format(P_ID)
        if Path(self.distro_path).is_file():
            self.distro_dict = io.loadmat(self.distro_path)
        else:
            self.form_distro()
            
        self.RR_distro=self.distro_dict['RR_distro']
        self.HR_clusters=(self.distro_dict['HR_clusters']).reshape(-1)
        
    def form_distro(self):
        '''
        TODO
        Given data, estimate the distro and save 
        '''
        pass

            
# =============================================================================
#     def sample_RR(self,HR,RR_prev):
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

    def sample_RR(self,HR,RR_prev):
        list_pdf_RR_joint=[self.RR_distro[j,0] for j in 
                           range(len(self.RR_distro))]
        list_pdf_RR_row_sum=[np.sum(arr,axis=0) for arr in list_pdf_RR_joint]
        #get cluster
        HR_up=(self.HR_clusters>HR).astype(int)
        #print(HR_up.shape,(np.diff(HR_up)).astype(bool))
        z=(np.arange(len(self.HR_clusters)-1))[(np.diff(HR_up)).astype(bool)][0]
        #get distros
        RR_z_distro=list_pdf_RR_row_sum[z]
        RR_z_vals=self.RR_distro[z,1].reshape(-1)
        #sample
        idx_Rn=np.random.choice(len(RR_z_vals),p=RR_z_distro) #sample RR_next idx
        RR_next=RR_z_vals[idx_Rn]
        return RR_next
    
    def __call__(self,HR_curve,Fs_out=25):
        factr=int(Fs_out/self.Fs_HR)
        arr_pk=np.zeros(factr*len(HR_curve))
        #TODO: bunch of changes here
        #gauss=norm(loc = 0., scale = 1.5).pdf(np.arange(-3,3+1))
        #plt.figure();plt.plot(gauss)
        RR_prev=0
        i=int(self.Fs_HR*self.w_l)
        w_r=int(((self.w_pk-self.w_l)*self.Fs_HR))-1
        while i < (len(HR_curve)-w_r-1):
            #get next RR
            arr_pk[factr*i:factr*i+1]=1
            #get next RR_interval
            #avg_HR=np.mean(HR_curve1[i-w_l:i+w_r+1])
            avg_HR=np.mean(HR_curve[i+w_r+1:i+w_r+1+self.Fs_HR]) #look ahead HR
            RR_next=self.sample_RR(avg_HR,RR_prev)
            i+=RR_next
        return arr_pk
#%% Sample Client
if __name__=='__main__':
    import pandas as pd
    import glob
    from data.sim_for_model_4 import HR_func_generator
    from HR2ecg_ppg_Simulator import filtr_HR,arr_t
    
    def Rpeak2HR(test_pks,win_len=10*100,step=1,Fs_ecg=100):
        HR_curve=[np.sum(test_pks[step*i:step*i+win_len])/(Fs_ecg/100) for i in 
              range(int((len(test_pks)-win_len+1)/step))]
        HR_curve=filtr_HR(np.array(HR_curve[::4])*60)
        return HR_curve
    
    def Rpeaks2RRint(arr_pks, Fs_pks=100):
        r_pk_locs_origin=np.arange(len(arr_pks))[arr_pks.astype(bool)]
        RR_ints=np.diff(r_pk_locs_origin/Fs_pks).reshape((-1,1))
        return RR_ints
    
    def get_train_data(path,val_files=[],test_files=[]):
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
        list_HR=[Rpeak2HR(arr_pks) for arr_pks in list_arr_pks]
        
        #select HR only at peak locations for training
        list_HR=[list_HR[j][list_arr_pks[j].astype(bool)] for j in 
                         range(len(list_HR))]
        #Get RR_ints in seconds
        list_RR_int=[Rpeaks2RRint(arr_pks) for arr_pks in list_arr_pks]
        list_RR_int_m1=[np.concatenate([RR_ints[0:1],RR_ints[:-1]],axis=0) 
                        for RR_ints in list_RR_int]
        list_HR=[HR[:-1] for HR in list_HR] #remove last HR as unusable

        return list_HR,list_arr_pks

    def get_test_data(file_path):
        df=pd.read_csv(file_path,header=None)
        arr=df.values
        test_out_for_check=arr[:,45:49].reshape(-1)
        test_in=Rpeak2HR(test_out_for_check)
        return test_in,test_out_for_check

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
    Fs=100
    t=arr_t[np.random.randint(len(arr_t))] # sample seq. length in s.
    t1=np.linspace(0,t,num=int(t*Fs),endpoint=False)
    HR_curve_f,D_HR=HR_func_generator(t1)
    
    #Use simulator to produce synthetic output given input
    
    test_pks,test_out_for_check,mean=get_test_data(test_files[0])
    Fs_ppg=25;Fs_ecg=100
    

    #plt.figure();plt.plot(HR_curve)
    #synth_ecg_out,test_in_ecg,synth_ppg_out,test_in_ppg=sim_pks2sigs(test_in)
    
    #define constants
    path_prefix=  'E:/Box Sync/' #'C:/Users/agarwal.270/Box/'
    path=path_prefix+'SP20/Research/PPG_ECG_proj/simulator_CC/data/'
    P_ID='W'
    Fs=25 #Hz
    
    #generate HR
    len_in_s=20.48 #s
    len_out=4
    len_in=Fs*len_in_s
    arr_t=np.arange(250,900,len_in_s) #change time duration when longer noise exists
    t=arr_t[np.random.randint(len(arr_t))] # sample seq. length in s.
    t1=np.linspace(0,t,num=int(t*Fs),endpoint=False)
    HR_curve_f,D_HR=HR_func_generator(t1)
    
    #Test Simulator
    sim_HR2pks=HR2Rpeaks_Simulator([],[],path=path,P_ID=P_ID,Fs_HR=Fs)
    arr_pk=sim_HR2pks(HR_curve_f)
    
    #Check if upsampling works
    arr_pk_upsampled=sim_HR2pks(HR_curve_f,Fs_out=100)
    check=arr_pk_upsampled.reshape(-1,4)
    plt.figure();plt.plot(arr_pk);plt.plot(arr_pk_upsampled[::4])
    #ppg1,HR1=gen_ppg_from_HR(t1,HR_curve_f,D_HR,peak_id,make_plots=make_plots)