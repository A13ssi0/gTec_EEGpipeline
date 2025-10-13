#!/usr/bin/env python

from fgmdm_riemann.fgmdm_riemann import FgMDM
import numpy as np
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.test import is_sym_pos_def
from scipy.io import loadmat, savemat
from py_utils.signal_processing import get_bandranges, get_trNorm_covariance_matrix, logbandpower
from py_utils.data_managment import get_files, save
from py_utils.eeg_managment import select_channels,get_EventsVector_onFeedback,get_channelsMask
from riemann_utils.covariances import get_riemann_mean_covariance, center_covariances
from datetime import datetime
import os



def main(filter_order=2, windowsLength=1, applyLaplacian=False, classes=None):

    bandPass = [[6, 24]]
    stopBand = [[14, 18]]

    wantedChannels = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']

    rejectionThreshold = 0.51

    applyLog = False
    doRecenter = False

    genPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

    pathData = os.path.join(genPath,'recordings')
    pathModels = os.path.join(genPath,'models')

    logWindowLength = 1


    ## -----------------------------------------------------------------------------
    print(' - Loading and preprocessing files')
    [signal, events_dataFrame, h, fileNames] = get_files(pathData)
    events_dataFrame.columns = [col.lower() for col in events_dataFrame.columns]
    channels = [ch.replace (" ", "") for ch in h['channels']]
    fs = h['SampleRate']
    windowsShift = h['dataChunkSize']/fs
    subjectCode = fileNames[0].split('/')[-3]

    if 'bfbh' in fileNames[0]:   
        task = 'mi_bfbh'
        if classes is None: classes = [771, 773]
    elif 'lhrh' in fileNames[0]: 
        task = 'mi_lhrh'
        if classes is None: classes = [769, 770]


    #-------------------------------------------------------------------------------
    laplacian = np.eye(signal.shape[-1])
    if applyLaplacian:
        pathLaplacian = None
        print(' - Applying Laplacian')   
        if h['device'].startswith('NA'):    pathLaplacian = os.path.join(pathData, 'lapMask16Nautilus.mat')
        elif h['device'].startswith('UN'):  pathLaplacian = os.path.join(pathData, 'lapMask8Unicorn.mat')

        if pathLaplacian is not None :
            laplacian = loadmat(pathLaplacian)
            laplacian = laplacian['lapMask']
        else:           
            print(f" - WARNING: Device {h['device']} not recognized. No Laplacian will be applied.")                    
               
    
    lap_signal = signal @ laplacian
   

    # ## ----------------------------------------------------------------------------- Processing Train
    filt_signal = lap_signal
    if len(bandPass)>0: filt_signal = get_bandranges(filt_signal, bandPass, fs, filter_order, 'bandpass')
    if len(stopBand)>0: filt_signal = get_bandranges(filt_signal, stopBand, fs, filter_order, 'bandstop')

    if applyLog: filt_signal = logbandpower(filt_signal, fs, slidingWindowLength=logWindowLength)  



    # ## -----------------------------------------------------------------------------    
    wantedChannels = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']
    # wantedChannels = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'O1', 'Oz', 'O2']
    channelMask = get_channelsMask(wantedChannels, channels)
    filt_signal = filt_signal[:, :, channelMask]
    # channels = ['Fz', 'C3', 'Cz', 'C4', 'Pz', 'PO7', 'Oz', 'PO8']


    # ## ----------------------------------------------------------------------------- Covariances
    [covs, cov_events] = get_trNorm_covariance_matrix(filt_signal, events_dataFrame, windowsLength, windowsShift, fs)
    labelVector = get_EventsVector_onFeedback(cov_events, covs.shape[1], classes)
    fdbVector = np.isin(labelVector, classes)

    covs_centered = covs
    if doRecenter:
        # Recenter cov matrices, reference = Riemannian mean of training set only 
        mean_cov, _ = get_riemann_mean_covariance(covs[:, fdbVector])
        inv_sqrt_mean_cov = np.expand_dims(invsqrtm(mean_cov), 1)
        print(' - Recentering covariance matrices around eye')
        covs_centered = center_covariances(covs, mean_cov, inv_sqrt_mean_cov)
        

    # if not (is_sym_pos_def(covs_centered[:,fdbVector])): 
    if not (is_sym_pos_def(covs_centered)): 
        counter = 0
        for i in range(covs_centered.shape[1]):
            eigenvalues = np.linalg.eigvalsh(covs_centered[:,i])
            if np.any(eigenvalues <= 0):    counter += 1
        raise ValueError(f' -ERROR : {counter} covariance matrices are not full rank of a total of {covs_centered.shape[1]}.')

    
    # ## ----------------------------------------------------------------------------- Fitting models
    covs_centered = covs_centered[:, fdbVector]
    labelVector = labelVector[fdbVector]
    
    print(f' - Training models with {covs_centered.shape}')
    fgmdm = FgMDM(njobs=-1)
    fgmdm.train(covs_centered, labelVector, classes, rejectionTh=rejectionThreshold)

    model = {
        'fgmdm': fgmdm,
        'mean_cov': mean_cov if doRecenter else None,
        'bandPass': bandPass,
        'stopBand': stopBand,
        'fs': fs,
        'filter_order': filter_order,
        'windowsLength': windowsLength,
        'windowsShift': windowsShift,
        'inv_sqrt_mean_cov': inv_sqrt_mean_cov if doRecenter else None,
        'classes': classes,
        'channels': channels,
        'trainFiles': fileNames,
        'laplacian': laplacian,
        'rejectionThreshold': rejectionThreshold,
        'applyLog': applyLog,
        'logWindowLength': logWindowLength,
    }

    now = datetime.now().strftime("%Y%m%d.%H%M")

    model_dir = os.path.join(pathModels, subjectCode)
    if not os.path.exists(model_dir):   os.makedirs(model_dir)
    filename = f'{model_dir}/{subjectCode}.{now}.{task}'
    existing_files = [f for f in os.listdir(model_dir) if f.startswith(os.path.basename(filename))]
    if existing_files:  
        print(f"Warning: Files starting with {subjectCode}.{now}.{task} already exist. Adding a suffix to avoid overwriting.")
        filename += f'.{len(existing_files)}'

    # savemat(f'{filename}.joblib', model)
    save(filename, model)
    

if __name__ == '__main__':
    main()

    