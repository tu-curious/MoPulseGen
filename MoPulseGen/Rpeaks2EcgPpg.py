# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 20:47:57 2020

@author: agarwal.270a
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import io
from scipy import signal as sig
from scipy.signal import detrend
from pathlib import Path
from simulator_for_CC import Simulator
import tensorflow as tf
import datetime
#%%
class Rpeaks2EcgPpg_Simulator(Simulator):
    '''
    find peak_train. from there on, more or less ppg ecg
    '''
    def __init__(self,input_list,output_list,w_pk=1,w_l=0.4,P_ID='',path='./',
                 ppg_id='ppg_green'):
        '''
        in/output format: list of numpy arrays. 1st channel of output is 
        w_pk and w_l defined in seconds
        '''
        super(Rpeaks2EcgPpg_Simulator,self).__init__(input_list,output_list)
        # TODO: Didn't change w_pk and w_l sizes for ECG as ECG needs smaller 
        # windows than ppg. But might need to change depending upon changing Fs
        self.w_pk=w_pk
        self.w_l=w_l
        self.P_ID=P_ID
        self.path=path
        self.ppg_id=ppg_id
        self.ecg_id='ecg'
        self.Cout_ppg=(0,1)
        self.Cout_ecg=(1,5)
        self.Fs_ppg=25
        self.Fs_ecg=100
        self.Fs_pks=100
        
        # Find ppg_basis_dict as apt ppg_basis to randomly generate PPG peaks, 
        # arranged in decreasing order of prominence
        self.ppg_basis_path=path+"{}_{}_ppg_basis.mat".format(self.P_ID,
                                                              self.ppg_id)
        if Path(self.ppg_basis_path).is_file():
            self.ppg_basis_dict = io.loadmat(self.ppg_basis_path)
        else:
            self.regen_ppg_basis()
            
        # Find ecg_basis_dict as apt ecg_basis to randomly generate ecg peaks, 
        # arranged in decreasing order of prominence
        self.ecg_basis_path=path+"{}_{}_ecg_basis.mat".format(self.P_ID,
                                                              self.ecg_id)
        if Path(self.ecg_basis_path).is_file():
            self.ecg_basis_dict = io.loadmat(self.ecg_basis_path)
        else:
            self.regen_ecg_basis()
        
    
    def ppg_filter(self,X0,filt=True):
        '''
        Band-pass filter multi-channel PPG signal X0
        '''
        nyq=self.Fs_ppg/2
        X1 = sig.detrend(X0,type='constant',axis=0); # Subtract mean
        if filt:
            b = sig.firls(219,np.array([0,0.3,0.5,4.5,5,nyq]),
                          np.array([0,0,1,1,0,0]),np.array([10,1,1]),nyq=nyq)
            X=np.zeros(X1.shape)
            for i in range(X1.shape[1]):
                #X[:,i] = sig.convolve(X1[:,i],b,mode='same'); # filtering using convolution, mode='same' returns the 'centered signal without any delay
                X[:,i] = sig.filtfilt(b,[1],X1[:,i])
    
        else:
            X=X1
        return X
    
    def regen_ppg_basis(self,save_flag=False):
        '''
        Regenerate the ppg_basis
        '''
# =============================================================================
#         self.ppg_basis_dict=self.extract_components(self.input,self.output,
#                                                 self.Cout_ppg,Fs_in=self.Fs_pks
#                                                 ,Fs_out=self.Fs_ppg,
#                                                 w_pk=self.w_pk,w_l=self.w_l,
#                                                 path4basis=self.ppg_basis_path,
#                                                 save_flag=save_flag)
# =============================================================================
        self.ppg_gen_model=self.learn_gen_model(self.input,self.output,
                                        self.Cout_ppg,Fs_in=self.Fs_pks
                                        ,Fs_out=self.Fs_ppg,
                                        w_pk=self.w_pk,w_l=self.w_l,
                                        path4basis=self.ppg_basis_path,
                                        save_flag=save_flag)
        return
    
    def regen_ecg_basis(self,save_flag=False):
        '''
        Regenerate the ecg_basis. Note we take every 4th element only starting
        from 0 as there is only one ecg per 4 channels of green ppg here.
        Change this as the output list changes
        '''
# =============================================================================
#         self.ecg_basis_dict=self.extract_components(self.input[::4],
#                                                 self.output[::4],
#                                                 self.Cout_ecg,Fs_in=self.Fs_pks
#                                                 ,Fs_out=self.Fs_ecg,
#                                                 w_pk=self.w_pk,w_l=self.w_l,
#                                                 path4basis=self.ecg_basis_path,
#                                                 save_flag=save_flag)
# =============================================================================
        self.ecg_gen_model=self.learn_gen_model(self.input[::4],
                                        self.output[::4],
                                        self.Cout_ecg,Fs_in=self.Fs_pks
                                        ,Fs_out=self.Fs_ecg,
                                        w_pk=self.w_pk*0.75,w_l=self.w_l*0.75,
                                        path4basis=self.ecg_basis_path,
                                        save_flag=save_flag)
        return
    
    def learn_gen_model(self,input_list,output_list,Cout,Fs_in,Fs_out,w_pk,
                           w_l,path4basis,save_flag=False,make_plots=True):
        '''
        Learn a generative model
        
        '''
        # Convert window lenghts to n_samples of output
        w_pk=int(w_pk*Fs_out)
        w_l=int(w_l*Fs_out)
        factr=int(Fs_in/Fs_out)
        
        list_r_pk_locs=[np.arange(len(arr_pks))[arr_pks.astype(bool)] for 
                        arr_pks in input_list]
        
        #??remove terminal pk_locs for simplified clipping for now
        list_r_pk_locs=[r_pk_locs[1:-1] for r_pk_locs in list_r_pk_locs]
        
        #get nearest dsampled idx
        list_r_pk_locs_dsampled=[np.floor(r_pk_locs/factr).astype(int) for 
                                 r_pk_locs in list_r_pk_locs]
        # find RR-intervals in seconds
        list_RR_ints=[np.diff(r_pk_locs/Fs_in) for r_pk_locs in list_r_pk_locs]
        list_RR_ints=[np.concatenate([RR_ints[0:1],RR_ints]) for RR_ints in 
                      list_RR_ints] #repeat the 1st element to avoid size mismatch
         
        list_wins_clean=[self.get_windows_at_peaks(r_pk_locs,
                            y[:,Cout[0]:Cout[1]].reshape(-1),
                            w_pk=w_pk,w_l=w_l,) for r_pk_locs,y in
                            zip(list_r_pk_locs_dsampled, output_list)]
        wins_clean=np.concatenate(list_wins_clean, axis=0)
        RR_int=np.concatenate(list_RR_ints, axis=0).reshape((-1,1))
        df=np.isnan(np.sum(np.sum(wins_clean,axis=1))) #check if any nan
        df2=np.isnan(np.sum(np.sum(RR_int,axis=1))) #check if any nan
        
        #shuffle
        perm=np.random.permutation(len(RR_int))
        RR_int,wins_clean=RR_int[perm],wins_clean[perm]
        
        #partition
        val_perc=0.14
        val_idx=int(val_perc*len(RR_int))
        val_data=[wins_clean[0:val_idx],RR_int[0:val_idx]]
        train_data=[wins_clean[val_idx:],RR_int[val_idx:]]

        
        def train_gen_model(train_data,val_data):
            from lib.model_CondVAE import Model_CondVAE
            #del Model_CondVAE
            optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
            #tensorboard stuff
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
            test_log_dir = 'logs/gradient_tape/' + current_time + '/test'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            test_summary_writer = tf.summary.create_file_writer(test_log_dir)
            
            # Create an instance of the model
            out_len=train_data[0].shape[1]
            h1=int(out_len/2)
            model = Model_CondVAE(optimizer = optimizer,
                                n_units=[h1,2,h1,out_len],
                                in_shape=train_data[0].shape[1],
                                cond_shape=train_data[1].shape[1])
            
            EPOCHS = 200
            #arr_K=np.arange(100,19,-10)
            #arr_epochs=np.arange(0,81,10)
            model.fit(data=[train_data,val_data],
                      summaries=[train_summary_writer,test_summary_writer],epochs = EPOCHS)
            return model
        
        model=train_gen_model(train_data,val_data)
        
        #val_data=[wins_clean[0:val_idx],RR_int[0:val_idx]]
        #x,y,mu,logsig,x_hat=model.predict(val_data)
        
        if make_plots:
            self.make_gen_plots(model)
            #plt.close('all')
# =============================================================================
#             for idx in range(10):
#                 plt.figure()
#                 for a in range(1,5):
#                     plt.subplot(2,2,a);plt.plot(x[a*idx]);plt.plot(x_hat[a*idx])
#                     plt.title('True and synthetic sample for RR={}'.format(y[a*idx]))
#                     plt.legend(['True','Synthetic']);plt.grid(True);a+=1
# =============================================================================
            
        return model
    
    def make_gen_plots(self,model):
                        
        # gen samples at a fixed RR interval
        RR = 1
        y_vec=np.array(RR).reshape((1,1))
        #y_vec=np.zeros((1,10))
        #y_vec[:,dig]=1
        
        sides = 5
        max_z = 4
        img_it = 0
        list_z_vals=np.round(np.linspace(-max_z,max_z,num=sides),1)
        plt.figure()
        for i in range(0, sides):
            z1 = list_z_vals[i] #(((i / (sides-1)) * max_z)*2) - max_z
            for j in range(0, sides):
                z2 = list_z_vals[j] #(((j / (sides-1)) * max_z)*2) - max_z
                z_ = np.array([z1, z2]).reshape((1,-1))
                decoded = model.decoder(z_,y_vec).numpy()
                plt.subplot(sides, sides, 1 + img_it)
                img_it +=1
                plt.plot(decoded.reshape(-1));plt.grid(True)
                plt.title('z1={}, z2={}'.format(z1,z2))
                #plt.axis('off')
                
        #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=.2)
        plt.suptitle('Peaks at RR={}'.format(y_vec[0,0]))
            
        # gen samples at different RR intervals
        sides = 5
        max_z = 4
        img_it = 0
        list_z_vals=np.round(np.linspace(-max_z,max_z,num=sides),0)
        list_RR_vals=np.round(np.linspace(0.5,1.5,num=sides),1)

        plt.figure()
        for i in range(0, sides):
            z1 = list_z_vals[i] #(((i / (sides-1)) * max_z)*2) - max_z
            z2=1*z1
            for j in range(0, sides):
                y_vec = list_RR_vals[j].reshape((1,1))
                z_ = np.array([z1, z2]).reshape((1,-1))
                decoded = model.decoder(z_,y_vec).numpy()
                plt.subplot(sides, sides, 1 + img_it)
                img_it +=1
                plt.plot(decoded.reshape(-1));plt.grid(True)
                plt.title('RR={}, z1=z2={}'.format(int(y_vec[0,0]),int(z1)))
                #plt.axis('off')
        #plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=.2)
        
        n_plots=5
        list_RR_vals=np.round(np.linspace(0.5,1.5,num=n_plots),1)
        plt.figure()
        z1,z2=0,0
        y_vec = list_RR_vals.reshape((-1,1))
        z_ = np.array(n_plots*[z1, z2]).reshape((-1,2)).astype(np.float64)
        decoded = model.decoder(z_,y_vec).numpy()
        plt.plot(decoded.T)
        plt.legend(['RR={}'.format(rr) for rr in list_RR_vals])
        
        return
        
    def extract_components(self,input_list,output_list,Cout,Fs_in,Fs_out,w_pk,
                           w_l,path4basis,save_flag=False,make_plots=True):
        '''
        extract templates from output sample_data using input sample_data and 
        find a template basis using PCA
        '''
        # Convert window lenghts to n_samples of output
        w_pk=int(w_pk*Fs_out)
        w_l=int(w_l*Fs_out)
        factr=int(Fs_in/Fs_out)
        
        list_r_pk_locs=[np.arange(len(arr_pks))[arr_pks.astype(bool)] for 
                        arr_pks in input_list]
        #get nearest dsampled idx
        list_r_pk_locs_dsampled=[np.floor(r_pk_locs/factr).astype(int) for 
                                 r_pk_locs in list_r_pk_locs]
        list_wins_clean=[self.get_windows_at_peaks(r_pk_locs,
                            y[:,Cout[0]:Cout[1]].reshape(-1),
                            w_pk=w_pk,w_l=w_l,) for r_pk_locs,y in
                            zip(list_r_pk_locs_dsampled, output_list)]
        wins_clean=np.concatenate(list_wins_clean, axis=0)
        #print(list_wins_clean[0].shape,wins_clean.shape)

        mat1=self.remove_outliers(wins_clean)
        #find eig values and observe
        eigen_vals1,eigen_vecs1,avg1=self.pca(mat1) 
        
        if make_plots:
            plt.figure();plt.subplot(121);plt.plot(eigen_vals1)
            plt.subplot(122);plt.plot(avg1)
            j=0
            while j<15:
                plt.figure();k=1
                while ((k<=8) and (j+k)<len(eigen_vals1)):
                    plt.subplot(4,2,k)
                    cntr=j+k;
                    plt.plot(eigen_vecs1[:,cntr],'r')
                    plt.grid(True)
                    plt.title('eig_peak '.format(cntr))
                    k=k+1;
                j=j+8;
                
        #store basis
        basis_dict={'eig_val':eigen_vals1,'eig_vec':eigen_vecs1,'mean':avg1}
        if save_flag:
            io.savemat(path4basis,mdict=basis_dict)
            
        return basis_dict
    
    def __call__(self,arr_pks,k_ppg=10,k_ecg=25):
        '''
        arr_pks: Location of R-peaks determined from ECG
        k: No. of prominent basis to keep
        '''
        factr_ppg=int(self.Fs_pks/self.Fs_ppg)
        factr_ecg=int(self.Fs_pks/self.Fs_ecg)
        r_pk_locs=np.arange(len(arr_pks))[arr_pks.astype(bool)]

        def get_signal(factr,r_pk_locs_origin,basis_dict,filt=False,
                       filtr=None,w_pk=1,w_l=0.4,Fs_out=25,k=10):
            w_pk=int(w_pk*Fs_out)
            w_l=int(w_l*Fs_out)
            w_r=w_pk-w_l-1
            #get nearest dsampled idx
            r_pk_locs=np.floor(r_pk_locs_origin/factr).astype(int)
            #remove terminal pk_locs
            r_pk_locs=r_pk_locs[(r_pk_locs>=w_l) & 
                            (r_pk_locs<=(int(len(arr_pks)/factr)-w_r-1))]
            
            n_peaks=int(len(arr_pks)/(factr*5))
            #sample bunch of peaks using PCA components
            eig_vec=basis_dict['eig_vec']
            eig_val=basis_dict['eig_val'].reshape((-1,1))
            avg=basis_dict['mean'].reshape((-1,1))
            eig_vec=eig_vec[:,:k];eig_val=eig_val[:k]
            l_peaks,n_coeff=eig_vec.shape
            weights=np.random.random_sample((n_coeff,n_peaks))*(eig_val**0.5)
            rand_pks=np.matmul(eig_vec,weights)+avg #form peaks
            
            #construct dsampled arr_pks
            arr_pks_dsampled=np.zeros(int(len(arr_pks)/factr))
            arr_pks_dsampled[r_pk_locs]=1
            #construct arr_ppg
            arr_y=np.zeros(int(len(arr_pks)/factr))
            #arr_pk=np.zeros(len(HR_curve1))
            #TODO: bunch of changes here
            #gauss=norm(loc = 0., scale = 1.5).pdf(np.arange(-3,3+1))
            #PTT=np.random.randint(4,8) #sample a PTT value
            #plt.figure();plt.plot(gauss)
            #print(np.max(r_pk_locs),len(arr_ppg))
            
            #Place sampled ppg peaks
            for i in range(len(r_pk_locs)):
                arr_y[r_pk_locs[i]-w_l:r_pk_locs[i]+w_r+1]+=rand_pks[:,i]
                                                                    
            if filt:
                arr_y_filt=filtr(arr_y.reshape(-1,1))
            else:
                arr_y_filt=arr_y*1
            #TODO: Better stitching needed than simple LPF
            #arr_ppg_filt=self.ppg_filter(arr_ppg.reshape(-1,1))
            #plt.figure();plt.plot(arr_ppg);plt.plot(arr_ppg_filt,'g--')
            #plt.plot(r_pk_locs,arr_ppg[r_pk_locs],'r+')
            return arr_y_filt.reshape(-1),arr_pks_dsampled
        
        arr_ppg,arr_pks_ppg=get_signal(factr_ppg,r_pk_locs,self.ppg_basis_dict,
                                       filt=True,filtr=self.ppg_filter,
                                       w_pk=self.w_pk,w_l=self.w_l,
                                       Fs_out=self.Fs_ppg,k=k_ppg)
        arr_ecg,arr_pks_ecg=get_signal(factr_ecg,r_pk_locs,self.ecg_basis_dict,
                                       w_pk=self.w_pk*0.75,w_l=self.w_l*0.75,
                                       Fs_out=self.Fs_ecg,k=k_ecg)
        return arr_ecg,arr_pks_ecg,arr_ppg,arr_pks_ppg
    
#%% Client
if __name__=='__main__':
    # Data Helpers
    import glob
    import pandas as pd
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
                    #arr[:,41:45]=(detrend(arr[:,41:45].reshape(-1),0,'constant')
                     #               ).reshape((-1,4))
                    arr[:,41:45]=(arr[:,41:45]/np.mean(arr[:,41:45].reshape(-1)
                                    ))-1

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
        #arr[:,41:45]=(detrend(arr[:,41:45].reshape(-1),0,'constant')).reshape((-1,4))
        arr[:,41:45]=(arr[:,41:45]/np.mean(arr[:,41:45].reshape(-1)))-1

        test_out_for_check=[np.concatenate([arr[:,29:30],arr[:,41:45]],axis=-1),
                            arr[:,30:31],arr[:,39:40],arr[:,40:41]]
        test_in=arr[:,45:49].reshape(-1)
        return test_in,test_out_for_check
    
    #Get Train Data for simulator
    plt.close('all')
    path_prefix= 'E:/Box Sync/' #'C:/Users/agarwal.270/Box/' #
    path=(path_prefix+'AU19/Research/PPG_ECG_proj/data/Wen_data_28_Sep/'
          'clean_lrsynced\\')
    val_files=[path+'2019092801_3154_clean.csv']
    test_files=[path+'2019092820_5701_clean.csv']
    input_list,output_list=get_train_data(path,val_files,test_files)
    
    #Create Simulator using train data
    sim_pks2sigs=Rpeaks2EcgPpg_Simulator(input_list,output_list,P_ID='W',
            path='E:/Box Sync/SP20/Research/PPG_ECG_proj/simulator_CC/data/')
    
    n_plots=5
    list_RR_vals=np.round(np.linspace(0.5,1.5,num=n_plots),1)
    z1,z2=0,0
    y_vec = list_RR_vals.reshape((-1,1))
    z_ = np.array(n_plots*[z1, z2]).reshape((-1,2)).astype(np.float64)
    decoded = sim_pks2sigs.ppg_gen_model.decoder(z_,y_vec).numpy()
    plt.figure()
    plt.plot(decoded.T)
    plt.legend(['RR={}'.format(rr) for rr in list_RR_vals])
    decoded = sim_pks2sigs.ecg_gen_model.decoder(z_,y_vec).numpy()
    plt.figure()
    plt.plot(decoded.T)
    plt.legend(['RR={}'.format(rr) for rr in list_RR_vals])
    #learn_gen_model(self,input_list,output_list,Cout,Fs_in=100,Fs_out=100,w_pk=1,
     #                      w_l=0.4,path4basis,save_flag=False,make_plots=True):
    
    
    #Use simulator to produce synthetic output given input
    test_in,test_out_for_check=get_test_data(val_files[0])
    synth_ecg_out,test_in_ecg,synth_ppg_out,test_in_ppg=sim_pks2sigs(test_in)
    
    #Visualize
    plt.figure()
    plt.plot(test_out_for_check[0][:,0:1])
    plt.plot(synth_ppg_out)
    plt.plot(test_in_ppg)
    plt.legend(['True','Synthetic','R-peaks'])
    plt.grid(True)
    plt.figure()
    plt.plot(test_out_for_check[0][:,1:5].reshape(-1))
    plt.plot(synth_ecg_out)
    plt.plot(test_in_ecg)
    plt.legend(['True','Synthetic','R-peaks'])
    plt.grid(True)