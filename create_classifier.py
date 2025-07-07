#!/usr/bin/env python

from fgmdm_riemann.fgmdm_riemann import FgMDM
import numpy as np
from pyriemann.utils.base import invsqrtm
from pyriemann.utils.test import is_sym_pos_def
from scipy.io import loadmat
from py_utils.signal_processing import get_bandranges, get_trNorm_covariance_matrix
from py_utils.data_managment import get_files, save
from py_utils.eeg_managment import select_channels,get_EventsVector_onFeedback
from riemann_utils.covariances import get_riemann_mean_covariance, center_covariances
from datetime import datetime
import os



def main(task='mi_bfbh', filter_order=2, windowsLength=1, classes=[771, 773]):

    bandPass = [[6, 24]]
    stopBand = [[14, 18]]

    applyLog = True
    doRecenter = False

    pathModels = 'c:/Users/aless/Desktop/gNautilus/data/models'
    pathData = 'c:/Users/aless/Desktop/gNautilus/data/recordings'

    ## -----------------------------------------------------------------------------
    print(' - Loading and preprocessing files')
    [signal, events_dataFrame, h, fileNames] = get_files(pathData)
    events_dataFrame.columns = [col.lower() for col in events_dataFrame.columns]
    channels = h['channels']
    fs = h['SampleRate']
    windowsShift = h['dataChunkSize']/fs
    subjectCode = fileNames[0].split('/')[-3]


    # ## -----------------------------------------------------------------------------    
    wantedChannels = channels
    [lap_signal, channels] = select_channels(lap_signal, wantedChannels, actualChannels=channels)


    # ## ----------------------------------------------------------------------------- Processing Train
    # filters = [ [] for _ in range(max(len(bandPass), len(stopBand))) ]
    # for k,band in enumerate(bandPass):  filters[k].append(RealTimeButterFilter(order=filter_order, cutoff=band, fs=fs, type='bandpass'))
    # for k,band in enumerate(stopBand): filters[k].append(RealTimeButterFilter(order=filter_order, cutoff=band, fs=fs, type='bandstop'))
    filt_signal = lap_signal
    if len(bandPass)>0: filt_signal = get_bandranges(filt_signal, bandPass, fs, filter_order, 'bandpass')
    if len(stopBand)>0: filt_signal = get_bandranges(filt_signal, stopBand, fs, filter_order, 'bandstop')

    # if applyLog: filt_signal = utils.get_logpower(filt, fs)  # da sistemare ----------------------------------------------------------------------------------

    #-------------------------------------------------------------------------------   
    pathLaplacian = 'c:/Users/aless/Desktop/gNautilus/lapMask16Nautilus.mat'
    laplacian = loadmat(pathLaplacian)
    laplacian = laplacian['lapMask']
    lap_signal = signal @ laplacian

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
        

    if not (is_sym_pos_def(covs_centered)): raise ValueError('Covariance matrices are not symmetric positive definite')

    
    # ## ----------------------------------------------------------------------------- Fitting models
    
    print(' - Training models')
    fgmdm = FgMDM(njobs=-1)
    fgmdm.train(covs_centered, labelVector, classes)

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
        'trainFiles': fileNames
    }

    now = datetime.now().strftime("%Y%m%d.%H%M")

    model_dir = os.path.join(pathModels, subjectCode)
    if not os.path.exists(model_dir):   os.makedirs(model_dir)
    filename = f'{model_dir}/{subjectCode}.{now}.{task}'
    existing_files = [f for f in os.listdir(model_dir) if f.startswith(os.path.basename(filename))]
    if existing_files:  
        print(f"Warning: Files starting with {subjectCode}.{now}.{task} already exist. Adding a suffix to avoid overwriting.")
        filename += f'.{len(existing_files)}'

    save(filename, model)
    

if __name__ == '__main__':
    main()

    