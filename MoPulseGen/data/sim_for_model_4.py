# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 15:00:21 2019

@author: agarwal.270a
"""
# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
from scipy.signal import windows as win
import pandas as pd
from scipy import io
import pickle
from scipy.stats import norm
# Import CC functions
#from cerebralcortex.core.datatypes import DataStream
#from cerebralcortex.core.metadata_manager.stream.metadata import Metadata, DataDescriptor, ModuleMetadata
#from cerebralcortex.core.util.spark_helper import get_or_create_sc


# Declare constants and load data
Fs=25 #Hz
len_in_s=20.48 #s
len_out=4
len_in=Fs*len_in_s
#arr_t=np.arange(250,290,len_in_s) #change time duration when longer noise exists
arr_t=np.arange(250,900,len_in_s) #change time duration when longer noise exists

path_prefix=  'E:/Box Sync/' #'C:/Users/agarwal.270/Box/'
path=path_prefix+'SU19/Research/PPG_ECG_Proj/py_code/MA_function/'
mdict=pickle.load(open(path+'data/sim_data.dat','rb'))
RR_distro=mdict['RR_distro']
HR_clusters=mdict['HR_clusters']

del mdict
#peak_data=mdict['peaks']
#led_id=mdict['led_id']

#verify after meeting
list_pdf_RR_joint=[RR_distro[j,0] for j in range(len(RR_distro))]
list_pdf_RR_row_sum=[np.sum(arr,axis=0) for arr in list_pdf_RR_joint]
list_pdf_RR_col_sum=[np.sum(arr,axis=1) for arr in list_pdf_RR_joint]
diff_arr=np.array([np.linalg.norm(list_pdf_RR_row_sum[k]-list_pdf_RR_col_sum[k])\
                   for k in range(len(list_pdf_RR_row_sum))]).round(4)
# =============================================================================
# plt.figure();
# for j in range(len(list_pdf_RR_row_sum)):
#     plt.subplot(7,2,j+1);plt.plot(list_pdf_RR_row_sum[j],'b-o')
#     plt.plot(list_pdf_RR_col_sum[j],'r--x');plt.legend(['row','col'])
#     plt.grid(True);plt.title('z={}, rmse={}'.format(j+1,diff_arr[j]))
# 
# =============================================================================


#%% Helper funcs
# =============================================================================
# def sample_RR(HR,RR_prev):
#     #get cluster
#     HR_up=(HR_clusters>HR).astype(int)
#     z=(np.arange(len(HR_clusters)-1))[(np.diff(HR_up)).astype(bool)][0]
#     #RR_z=RR_distro[z]
#     RR_z_distro=RR_distro[z,0];RR_z_vals=RR_distro[z,1].reshape(-1)
#     if RR_prev==0: #beginning of sampling. sample uniform randomly
#         RR_next=RR_z_vals[np.random.randint(len(RR_z_vals))]
#     else:
#         idx_Rp=np.arange(len(RR_z_vals))[RR_z_vals==RR_prev]
#         RR_z_Rp=RR_z_distro[idx_Rp,:] #conditional distro given z, RR_p
#         idx_Rn=np.random.choice(len(RR_z_vals),p=RR_z_Rp/np.sum(RR_z_Rp)) #sample RR_next idx
#         RR_next=RR_z_vals[idx_Rn]
#     return RR_next
# =============================================================================

def sample_RR(HR,RR_prev):
    #get cluster
    HR_up=(HR_clusters>HR).astype(int)
    z=(np.arange(len(HR_clusters)-1))[(np.diff(HR_up)).astype(bool)][0]
    #get distros
    RR_z_distro=list_pdf_RR_row_sum[z]
    RR_z_vals=RR_distro[z,1].reshape(-1)
    #sample
    idx_Rn=np.random.choice(len(RR_z_vals),p=RR_z_distro) #sample RR_next idx
    RR_next=RR_z_vals[idx_Rn]
    return RR_next

def sinusoid(t,w,phi,Fs=25):
    '''
    Takes in inputs as numpy arrays of same size. Returns the sinewave with
    desired characteristics.
    t: array of time values in seconds. If a scalar is supplied, it is 
    considered as duration of the time series in seconds starting from 0. It is
    divided into t*Fs divisions.
    w: array of angular frequencies in radians/seconds. If a scalar is 
    supplied, it is made into a constant array of same shape as t and value w.
    phi: array of phase values in radians.  If a scalar is supplied, it is made
    into a constant array of same shape as t and value phi.
    Fs= Sampling frequency in Hz. Only needed in case t is not an array.
    returns: t, s=np.sin(w*t+phi)
    '''
    # Handle Scalar inputs
    if not(hasattr(t, "__len__")):
        t=np.linspace(0,t,num=t*Fs,endpoint=False)
    if not(hasattr(w, "__len__")):
        w=w*np.ones(t.shape)
    if not(hasattr(phi, "__len__")):
        phi=phi*np.ones(t.shape)
    # Check shapes are same
    if (w.shape!=t.shape and phi.shape!=t.shape):
        raise TypeError('Dimensional mismatch between input arrays. Please check the dimensions are same')
    s=np.sin(w*t+phi)
    return t,s

def HR_func_generator(t1):
    arr_HR=np.arange(50,180) # Possible heart rates
    
    # make a array of functions
    f1=lambda B,D:((D*win.triang(len(t1))).astype(int)+B).astype(np.float32) #triang
    f2=lambda B,D:((D*win.triang(2*len(t1))).astype(int)+B).astype(np.float32)\
    [:len(t1)] # 1st half of triang
    f3=lambda B,D:((D*win.tukey(len(t1),alpha=(0.3*np.random.rand()+0.7))).astype(int)+B).astype(np.float32) #tukey
    f4=lambda B,D:((D*win.tukey(2*len(t1),alpha=(0.3*np.random.rand()+0.7))).astype(int)+B)\
    .astype(np.float32)[:len(t1)] # 1st half of tukey
    arr_f=np.array(1*[f1]+1*[f2]+1*[f3]+1*[f4]) # possible to change the proportion of functions
    
    #randomly select elements
    D_HR=0;HRs=[];D_HR_max=50
    while D_HR==0: # we don't want D_HR to be zero so keep resampling
        HRs+=[arr_HR[np.random.randint(len(arr_HR))]]
        HR_range=np.arange(HRs[0]+1,min([HRs[0]+D_HR_max,180])+1)
        HRs+=[HR_range[np.random.randint(len(HR_range))]]
        B_HR,D_HR=HRs[0],HRs[1]-HRs[0]
    #B_HR,D_HR=arr_B_HR[np.random.randint(len(arr_B_HR))],arr_D_HR[np.random.randint(len(arr_D_HR))]
    HR_curve_f=arr_f[np.random.randint(len(arr_f))](B_HR,D_HR) #trend
    return HR_curve_f,D_HR

def filtr(X0,Fs=25,filt=True):
    nyq=Fs/2;flag=False
    
    if len(X0.shape)==1:
        X0=X0.reshape(-1,1)
        flag=True
    X1 = sig.detrend(X0,type='constant',axis=0); # Subtract mean
    if filt:
        # filter design used from Ju's code with slight changes for python syntax
        b = sig.firls(219,np.array([0,0.3,0.5,4.5,5,nyq]),np.array([0,0,1,1,0,0]),np.array([10,1,1]),nyq=nyq);
        X=np.zeros(X1.shape)
        for i in range(X1.shape[1]):
            #X[:,i] = sig.convolve(X1[:,i],b,mode='same'); # filtering using convolution, mode='same' returns the centered signal without any delay
            X[:,i] = sig.filtfilt(b, [1], X1[:,i])
    else:
        X=X1
        
    if flag:
        X=X.reshape(-1)
    #X=sig.detrend(X,type='constant',axis=0); # subtracted mean again to center around x=0 just in case things changed during filtering
    return X

def filtr_HR(X0,Fs=25,filt=True):
    nyq=Fs/2;flag=False
    
    if len(X0.shape)==1:
        X0=X0.reshape(-1,1)
        flag=True
    X1 = np.copy(X0)#sig.detrend(X0,type='constant',axis=0); # Subtract mean
    if filt:
        # filter design used from Ju's code with slight changes for python syntax
        b = sig.firls(219,np.array([0,0.5,1,nyq]),np.array([1,1,0,0]),np.array([1,1]),nyq=nyq);
        X=np.zeros(X1.shape)
        for i in range(X1.shape[1]):
            #X[:,i] = sig.convolve(X1[:,i],b,mode='same'); # filtering using convolution, mode='same' returns the centered signal without any delay
            X[:,i] = sig.filtfilt(b, [1], X1[:,i])
    else:
        X=X1
        
    if flag:
        X=X.reshape(-1)
    #X=sig.detrend(X,type='constant',axis=0); # subtracted mean again to center around x=0 just in case things changed during filtering
    return X

def normalize_AC(data_left_filt,Fn=25,c=0,make_plots=False):
    '''
    data_left_filt: filtered ppg data
    Fn: Sampling frequency in Hz
    c: Column (Channel) in the array to be normalized
    '''
    data_left_filt=1*data_left_filt
    flag=False
    if len(data_left_filt.shape)==1:
        data_left_filt=data_left_filt.reshape((-1,1))
        flag=True
        
    prc_l=50
    pk_idx_start=2*Fn;pk_idx_end=29*Fn;
    
    y=data_left_filt[pk_idx_start:pk_idx_end,c]
    locs,pk_props = sig.find_peaks(y,distance=8,height=0);
    pks_l=y[locs]
    locs=locs+pk_idx_start;
    
    if make_plots:
        plt.figure(); plt.subplot(211);
        plt.plot(data_left_filt[:pk_idx_end,c]);plt.plot(locs,pks_l,'r+')


    temp_mins_l=[];
    #for j=[-5,-4,-3,-2,-1,1,2,3,4,5]
    for j in range(-7,0):
        temp_mins_l+=[data_left_filt[locs+j,c]];
    temp_min_l=np.min(np.array(temp_mins_l),axis=0);
    amp_left=np.nanpercentile(pks_l-temp_min_l,prc_l);
    #amp_left=np.mean(pks_l-temp_min_l);

    data_left_filt[:,c]=data_left_filt[:,c]/amp_left;
    if flag:
        data_left_filt=data_left_filt.reshape(-1)     
    return data_left_filt

def form_data(X,Y,len_in,len_out):
    '''
    X:timeseries with inputs
    Y:timeseries with outputs
    '''
    in_size=int(len_in)
    out_size=int(len_out)
    step_size=int(len_out/4)#np.max([out_size,4]) #change this as desired
    
    #clip timeseries to nearest multiple of step_size
    #lenth1=(((len(X)-in_size)//step_size)*step_size)+in_size
    lenth=len(X)
    #print(lenth1,lenth)
    X,Y=X.T,Y.T # Transpose to make it look like time-series
    X,Y=X.reshape(X.shape+(1,)),Y.reshape(Y.shape+(1,)) # add a dimension for concatenation
    #print(X.shape,Y.shape)
    #idx=np.arange(0,lenth-in_size,step_size)+in_size
    idx=step_size*np.arange(0,1+((lenth-in_size)//step_size))+in_size
    #print(idx[-1])
    #print(lenth,X.shape[1],len(idx),(X.shape[1]-in_size+1)//step_size)
    #print(X.shape,Y.shape,HR.shape)
    data_X=np.concatenate([X[:,i-in_size:i,:] for i in idx],axis=-1).T
    data_Y=np.concatenate([Y[i-out_size:i,:] for i in idx],axis=-1).T
    #kernel_size=100;stride=1
    #idxHR=np.arange(i-out_size+kernel_size,i,stride)
    return data_X,data_Y

def pd_ffill(arr):
    df = pd.DataFrame(arr)
    df.fillna(method='ffill', axis=0, inplace=True)
    out = df.values.reshape(arr.shape)
    return out

def add_motion_noise(ppg1,flag=True):
    # Noise for SNR=10log10(P_s/P_n)=20 dB => sigma=(ppg_pow**0.5)/10
    acc1=0.00*np.random.standard_normal(ppg1.shape) # random normal noise with (0,0.1^2)
    if flag: #extreme motion artefacts to be added or not
        acc1=acc1+(2*np.random.random_sample(ppg1.shape)-1) # [-2,2] random uniform
    #f=lambda z: (3 / (1 + np.exp(-10*z))) # A saturating sigmoid
    f=lambda z: 2*np.tanh(2*z)
    ppg1=ppg1+f(acc1) #noise added making values [-2,2] or [-4,4] depending on mode
    return ppg1,acc1

def extract_rand_noise(noiz_list,lenth):
    '''
    noiz_list: Available components to choose from
    lenth: Desired length of the noise signal
    '''
    noiz_list=[n for n in noiz_list if len(n)>lenth]
    if len(noiz_list)==0:
        raise AssertionError('Please use a smaller duration of ppg.')
    noiz=noiz_list[np.random.randint(len(noiz_list))]
    idx_start=np.random.randint(len(noiz)-lenth)
    noiz=noiz[idx_start:idx_start+lenth]
    return noiz

def gen_ppg_from_HR(t1,HR_curve_f,D_HR,peak_id,make_plots=False):
    '''
    mode={0:basic sinusoid, 1:mixture of sinusoids, 2:mixture of sinusoids with
    a lot of motion artifacts}
    '''
    # Randomly insert consecutive Nan's and then ffill
    perc_change=5;cons_reps=len(t1)//(np.abs(D_HR*2))
    #idx=1+np.random.RandomState(seed=seed1).permutation(len(t1)-2-cons_reps)[:int((len(t1)-2)/cons_reps*perc_change/100)]
    idx=1+np.random.permutation(len(t1)-2-cons_reps)[:int((len(t1)-2)/cons_reps*perc_change/100)]
    try:
        idx=np.concatenate([np.arange(i,i+cons_reps) for i in idx])
        HR_curve_f[idx]=np.nan
        HR_curve1=pd_ffill(HR_curve_f)
    except ValueError:
        HR_curve1=1*HR_curve_f
    
    # TODO: Removed 0.1 Hz and 0.4 Hz in HRV
    #HRV_w1=2*np.pi*0.1;HRV_w2=2*np.pi*0.4
    #rand_mix=np.repeat(np.random.random_sample(1+(len(t1)//1500)),1500)[:len(t1)]
    #rand_mix=0.55
    #print(len(t1),rand_mix)
    #gain_list=np.array([0,1,2,2,1,1,1,1])
    #HR_curve1+=0.03*((rand_mix*sinusoid(t1,HRV_w1,phi=0)[-1])+\
     #           ((1-rand_mix)*sinusoid(t1,HRV_w2,phi=0)[-1]))#*gain_list[(300/HR_curve1).astype(int)]
    #plt.figure();plt.plot(t1,sinusoid(t1,HRV_w1,phi=0)[-1],t1,sinusoid(t1,HRV_w2,phi=0)[-1])
    #HR_curve1,_=add_motion_noise(HR_curve1,flag=False)
    #print(HR_curve1.shape,t1.shape)
# =============================================================================
#     w1=2*np.pi*(HR_curve1/60)
#     #phi_PTT=(0.5*np.pi)/(HR_curve1/60)
#     phi_PTT=0
#     _,ppg0=sinusoid(t1,w1,phi=phi_PTT)
#     
#     ppg1=ppg0*2
#     PTT=np.random.randint(4,6) #sample a PTT value
#     ppg1=np.concatenate([np.zeros(PTT),ppg1[:-1*PTT]])
#     
#     
#     # Peak Detection & check figure for its accuracy
#     #out = ecg.ecg(signal=ppg01, sampling_rate=25,show=False)
#     #ind=out['rpeaks']
#     #arr_peaks=np.zeros(len(ppg01));arr_peaks[ind]=1
#     #arr_peaks=(ppg01==np.max(ppg01)).astype(int)
#     ind,_=find_peaks(ppg1,distance=6,height=0.9)
# 
# =============================================================================
    w_l=12;w_pk=25;w_r=w_pk-w_l-1
    n_peaks=int(len(HR_curve1)/5)
    #remove terminal pk_locs
    #ind=ind[ind>=w_l]
    #ind=ind[ind<(len(ppg1)-w_r)]
    
    #sample bunch of peaks using PCA components
    path2base='E:/Box Sync/'+\
    'AU19/Research/PPG_ECG_proj/data/Wen_data_28_Sep/clean_lrsynced\\'
    base_dict = io.loadmat(path2base+"green_ppg_basis.mat")
    #base_dict=mdict[peak_id+'_G']['peaks']
    eig_vec=base_dict['eig_vec'];eig_val=base_dict['eig_val'].reshape((-1,1))
    avg=base_dict['mean'].reshape((-1,1))
    k=10;eig_vec=eig_vec[:,:k];eig_val=eig_val[:k]
    l_peaks,n_coeff=eig_vec.shape
    weights=np.random.random_sample((n_coeff,n_peaks))*(eig_val**0.5)
    rand_pks=np.matmul(eig_vec,weights)+avg #form peaks
    #rand_pks=rand_pks[int(l_peaks/2)-w_l:int(l_peaks/2)+w_r+1,:] #keep desired width
    
    #OR
    
# =============================================================================
#     # Sample peaks randomly from those available in peak_mat
#     peak_mat=peak_dict[peak_id];l_peaks=peak_mat.shape[0]
#     rand_pks_idx=np.random.randint(peak_mat.shape[1],size=n_peaks)
#     rand_pks=peak_mat[int(l_peaks/2)-w_l:int(l_peaks/2)+w_r+1,rand_pks_idx]
#     
# =============================================================================
    arr_ppg=np.zeros(len(HR_curve1))
    arr_pk=np.zeros(len(HR_curve1))
    #TODO: bunch of changes here
    gauss=norm(loc = 0., scale = 1.5).pdf(np.arange(-3,3+1))
    PTT=np.random.randint(4,8) #sample a PTT value
    #plt.figure();plt.plot(gauss)
    RR_prev=0;i=1*w_l;cntr=0
    while i < (len(HR_curve1)-w_r-1):
        #get next RR
        arr_ppg[i-w_l:i+w_r+1]+=rand_pks[:,cntr]
        arr_pk[i-3-PTT:i+3+1-PTT]=gauss
        #get next RR_interval
        #avg_HR=np.mean(HR_curve1[i-w_l:i+w_r+1])
        avg_HR=np.mean(HR_curve1[i+w_r+1:i+w_r+1+Fs]) #look ahead HR
        RR_next=sample_RR(avg_HR,RR_prev)
        i+=RR_next
        cntr+=1
    
    
# =============================================================================
#     #sample bunch of noise using PCA components
#     noise_dict=mdict[peak_id+'_G']['noise']
#     #DC_list=noise_dict['DC']
#     NP_list=noise_dict['NP']
#     P_list=noise_dict['P'];N_list=noise_dict['N']
#     # Randomly pick one element from each list
#     #DC=DC_list[np.random.randint(len(DC_list))]
#     NP=extract_rand_noise(NP_list,len(arr_ppg))
#     P=extract_rand_noise(P_list,len(arr_ppg))
#     N=extract_rand_noise(N_list,len(arr_ppg))
#     
#     #get random gains for noise signals
#     gain_NP=(1-0.5)*np.random.rand()+0.5 #in [0.5,1)
#     gain_P,gain_N=gain_NP*np.random.rand(2) # in [0,gain_NP)
#     #if j<2:
#      #   gain_NP,gain_P,gain_N=0,0,0
#     
#     arr_ppg+=(gain_NP*NP+gain_P*P+gain_N*N) #Add noise
#     #arr_ppg=arr_ppg[:,j]*DC
#     
#     
# 
#     #arr_ppg_norm=1*arr_ppg
#     #plt.figure();plt.plot(arr_ppg);plt.plot(arr_ppg_norm,'--')
#     #plt.legend(['actual','AC Normalized'])
#     #add motion noise
#     #ppg2,acc1=add_motion_noise(arr_ppg,False)
#     
#     
#     ppg2=1*arr_ppg
#     ppg2_filt=filtr(ppg2.reshape(-1,1),Fs=25)
#     # Normalize AC component
#     ppg2_filt=normalize_AC(ppg2_filt,make_plots=False)
# =============================================================================
    #TODO: Converted HR to Hz from BPM and made it smoother
    ppg2=1*arr_ppg
    ppg2_filt=filtr(ppg2.reshape(-1,1),Fs=25)
    HR_filt=filtr_HR(HR_curve1/60)
    #arr_pk_filt=filtr(arr_pk,Fs=25)
    #ppg2=((ppg2+2)/4) # normalize using min-max of [-2,2]
    #acc1=((acc1+1)/2) # normalize using min-max of [-2,2]
    
    #plots
    if make_plots:
        #plt.figure()
        #plt.psd(HR_curve1[-Fs*10:], NFFT=Fs*10, Fs=Fs,detrend='constant')
        plt.figure()
        ax1=plt.subplot(311);ax1.plot(t1,HR_filt)
        ax1.set_title('HR');plt.grid(True)
        #ax2=plt.subplot(412,sharex=ax1);ax2.plot(t1,ppg1,t1[ind],ppg1[ind],'r+')
        #ax2.set_title('PPG_clean with detected peaks');plt.grid(True)
        #ax3=plt.subplot(413,sharex=ax1);ax3.plot(t1,acc1)
        #ax3.set_title('Acc');plt.grid(True)
        ax3=plt.subplot(312,sharex=ax1);ax3.plot(t1,arr_pk)
        ax3.set_title('filtered peak train');plt.grid(True)
        ax4=plt.subplot(313,sharex=ax1);ax4.plot(t1,ppg2_filt)
        ax4.set_title('filtered_PPG');plt.grid(True)
    
    return ppg2_filt,HR_filt

    
#%% Main
def main(data_size=10000,for_test=False,make_plots=False,save_data=False):
    while(True):
        t=arr_t[np.random.randint(len(arr_t))] # sample seq. length in s.
        # form HR curve
        t1=np.linspace(0,t,num=t*Fs,endpoint=False)
        HR_curve_f,D_HR=HR_func_generator(t1)
        peak_id='white'
        ppg1,HR1=gen_ppg_from_HR(t1,HR_curve_f,D_HR,peak_id,make_plots=make_plots)
        #print(HR1.shape,ppg1.shape)
        
        len_in=Fs*len_in_s;len_out=1*len_in
        data_X,data_Y=form_data(ppg1,HR1,len_in=len_in,len_out=len_out)
        #test
        if for_test:
            if save_data:
                mdict={'ppg':ppg1,'HR':HR1}
                io.savemat('eig_peaks_s.mat',mdict=mdict)
            return ppg1,HR1,data_X,data_Y
            
        if 'dataset_X' not in locals():
            dataset_X,dataset_Y=data_X,data_Y
        else:
            dataset_X=np.concatenate([dataset_X,data_X],axis=0)
            dataset_Y=np.concatenate([dataset_Y,data_Y],axis=0)
                
        if (len(dataset_Y)>=data_size):
            break
    dataset_X=dataset_X[:data_size].astype(np.float32)
    dataset_Y=dataset_Y[:data_size].astype(np.float32)

    #separate
    ratio=0.1;cut_idx=int(ratio*len(dataset_X))
    val_data = (dataset_X[:cut_idx],dataset_Y[:cut_idx])
    train_data = (dataset_X[cut_idx:],dataset_Y[cut_idx:])
    
    #shuffle
    idx = np.random.permutation(cut_idx)
    val_data=(val_data[0][idx],val_data[1][idx])
    idx = np.random.permutation(len(dataset_Y)-cut_idx)
    train_data=(train_data[0][idx],train_data[1][idx])
    
    return train_data,val_data


if __name__=='__main__':
    plt.close('all')
    X,Y=main()