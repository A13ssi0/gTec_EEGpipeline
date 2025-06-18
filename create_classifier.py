#!/usr/bin/env python

from fgmdm_riemann import FgMDM
import numpy as np
from pyriemann.utils.base import invsqrtm
from datetime import datetime
from pyriemann.utils.test import is_sym_pos_def
from scipy.io import savemat, loadmat

from py_utils.data_managment import get_files

def main():

    subject = 'i4'
    task = 'bfbh'

    filter_order = 4

    windowsLength = 1
    windowsShift = 1/16
    
    #bandranges = [[7, 14], [18, 23]]
    #bandranges = [[8, 12], [20, 30]]
    #bandranges = [[7, 13]]
    #bandranges = [[8, 20]]
    bandranges = [[7, 23]]
    #stoprange = [13, 18]
    stoprange = []

    classes = [771, 773]

    percVal = 0 # %validation

    pathData = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    pathModels = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
    pathLaplacian = 'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'

    ## -----------------------------------------------------------------------------
    print(' - Loading and preprocessing files')
    [signal, events_dataFrame, fs, directory_path] = get_files(pathData)

    #-------------------------------------------------------------------------------   
    laplacian = loadmat(pathLaplacian)
    laplacian = laplacian['lapmask']
    train = signal @ laplacian

    ## -----------------------------------------------------------------------------    
    # if train.shape[1] == 32:
    #     #wantedChannels = ['FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'C3', 'C1', 'CZ', 'C2', 'C4', 'CP3', 'CP1', 'PZ','CP2', 'CP4']
    #     wantedChannels = ['C3', 'C4', 'C1', 'C2', 'CZ', 'CP2', 'CP4', 'CP1', 'CP3', 'PZ']
    # elif train.shape[1] == 16:
    #     #wantedChannels = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3','C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz','CP2', 'CP4']
    #     #wantedChannels = ['FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C3','C1', 'Cz', 'C2', 'C4', 'CP3', 'CP1', 'CPz','CP2', 'CP4']
    #     wantedChannels = ['C3', 'C4', 'C1', 'C2', 'CZ', 'CP2', 'CP4', 'CP1', 'CP3', 'PZ']
    #     # wantedChannels = ['C3', 'C4', 'C1', 'C2', 'CZ', 'CP2', 'CP4', 'CP1', 'CP3']
        

    # channelGroups = [['FC5', 'C3','CP5', 'FC3', 'C5','CP3'],
    #                  ['FC1', 'FC2', 'CZ', 'CP1', 'CP2', 'FCZ', 'C1',  'C2', 'PZ' ],
    #                  ['FC6', 'C4','CP6', 'FC4', 'CP4','C6']]
    channelGroups = []

    [train, channels] = utils.select_channels(train, wantedChannels)

    # Avg chs for each group
    #train = utils.apply_ROI_over_channels(train, channels, channelGroups)

    ## ----------------------------------------------------------------------------- Processing Train
    filt = utils.get_bandranges(train, bandranges, fs, filter_order, stoprange)
    logBP = utils.get_logpower(filt, fs)

    # Get regularized cov matrix for each window and each band
    # events [samples] --> events [windows]
    [covs, cov_events, _] = utils.get_trNorm_covariance_matrix(filt, train_events, windowsLength, windowsShift, fs)
    [logBP_covs, _, _] = utils.get_trNorm_covariance_matrix(logBP, train_events, windowsLength, windowsShift, fs)
    n_samples = covs.shape[1] # #windows

    # Label each cont feedback window as the previous cue type, leave nan otherwise
    labelVector = utils.get_EventsVector_onFeedback(cov_events, n_samples, classes)

    # Split trials, get train/val windows idx
    [idx_train, idx_val, idx_subtrain, idx_subval] = utils.get_indices_train_validation(cov_events, classes, percVal = percVal)

    # Recenter cov matrices, reference = Riemannian mean of training set only 
    # mean_cov, _ = utils.get_riemann_mean_covariance(covs[:, idx_train], labelVector[idx_train])
    # inv_sqrt_mean_cov = np.expand_dims(invsqrtm(mean_cov),1)
    # print(' - Recentering covariance matrices around eye')
    # covs_centered = utils.center_covariances(covs, mean_cov, inv_sqrt_mean_cov)

    covs_centered = covs
    logBP_covs_centered = logBP_covs

    if not (is_sym_pos_def(covs_centered) & is_sym_pos_def(logBP_covs_centered)):
        print('NOT SPD')
    

    ## ----------------------------------------------------------------------------- Fitting models
    fgmdm = FgMDM(njobs=-1)
    fgmdm_logBP = FgMDM(njobs=-1)

    # Training
    print(' - Training models')
    fgmdm.train(covs_centered, labelVector, classes, idx_train, idx_val)
    fgmdm_logBP.train(logBP_covs_centered, labelVector, classes, idx_train, idx_val)

    files = {
        'fgmdm': fgmdm,
        'fgmdm_logBP': fgmdm_logBP,
        # 'mean_cov': mean_cov,
        'bandranges': bandranges,
        'fs': fs,
        'filter_order': filter_order,
        'windowsLength': windowsLength,
        'windowsShift': windowsShift,
        'wantedChannels': wantedChannels,
        # 'inv_sqrt_mean_cov': inv_sqrt_mean_cov,
        'channelGroups': channelGroups,
        'classes': classes,
        'stoprange': stoprange,
        'files': filenames
    }
    today = datetime.now()
    today = today.strftime("%Y%m%d-%H%M")
    
    # Validation ???? (already in fgmdm.val_probabilities)
    pred_proba = fgmdm.predict_probabilities(covs_centered[:, idx_train])
    #pred_proba = fgmdm.predict_probabilities(covs_centered[:, idx_val])

    print('-Base model')
    print(fgmdm.train_accuracies)
    print('-logBP model')
    print(fgmdm_logBP.train_accuracies)

    #????????????????
    # a = fgmdm.merge_classifiers(pred_proba)
   
    b = fgmdm.merge_bands(pred_proba, labelVector[idx_val], idx_subtrain, idx_subval)
    
    utils.save(pathModels+subject+'_'+task+'_'+today, files)
    
    # savemat(pathMAT+ subject+'_'+today+'_classifier_dump.mat', {'classifier_covs': covs_centered[:, idx_train], 'classifier_labels': labelVector[idx_train], 'classifier_probs': fgmdm.train_probabilities})




if __name__ == '__main__':
    main()

    