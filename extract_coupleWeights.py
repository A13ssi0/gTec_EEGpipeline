import os
import numpy as np
from datetime import datetime
from scipy.io import loadmat
import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
from py_utils.signal_processing import get_bandranges, get_trNorm_covariance_matrix
from py_utils.data_managment import get_files
from py_utils.eeg_managment import select_channels,get_EventsVector_onFeedback
from riemann_utils.covariances import get_riemann_mean_covariance, center_covariances
from pyriemann.utils.base import invsqrtm


def extract_coupleWeights(subjects, gammaMI=0, gammaRest=0, classes=None, doSave=True):

    if classes is None:
        classes = [773, 771]

    # Load datasets and process them    
    genPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    recordingsPath = os.path.join(genPath, 'recordings')
    modelsPath = os.path.join(genPath, 'models')

    doSyncronization = False
    probabilities, events, string_subject = loadProcess_datasets(subjects, recordingsPath, modelsPath, doSyncronization)

    # # Variables for calc weights
    # doLengthNormalization = True
    # notLengthNormalization = False
    # withRest = True
    # noRest = False

    # gamma = [gammaMI, gammaRest]
    
    # # Calc weighted average with rest
    # print(' ---------------------------------------------------------')
    # print('- WEIGHTS -')

    # print('-- LOSS --')
    # weights = {'loss': {}}
    # weights['loss']['notNormalized'] = {}
    # weights['loss']['normalized'] = {}

    # weights['loss']['notNormalized']['withRest'] = calc_fusionWeights_crossentropy(probabilities, events, withRest, notLengthNormalization, classes, gamma)
    # print(' --- notNormalized.withRest : ')
    # print(weights['loss']['notNormalized']['withRest'])
    
    # weights['loss']['normalized']['withRest'] = calc_fusionWeights_crossentropy(probabilities, events, withRest, doLengthNormalization, classes, gamma)
    # print(' --- normalized.withRest : ')
    # print(weights['loss']['normalized']['withRest'])
    
    # # Calc weighted average without rest
    # weights['loss']['notNormalized']['withoutRest'] = calc_fusionWeights_crossentropy(probabilities, events, noRest, notLengthNormalization, classes, gamma)
    # print(' --- notNormalized.withoutRest : ')
    # print(weights['loss']['notNormalized']['withoutRest'])
    
    # weights['loss']['normalized']['withoutRest'] = calc_fusionWeights_crossentropy(probabilities, events, noRest, doLengthNormalization, classes, gamma)
    # print(' --- normalized.withoutRest : ')
    # print(weights['loss']['normalized']['withoutRest'])

    # # Calc weight with accuracy
    # weights['accuracy'] = calc_fusionWeights_accuracy(probabilities, events, classes)
    # print('-- ACCURACY :')
    # print(weights['accuracy'])

    # # Saving
    # savePath = '/home/alessio/cBCI_ws/extra/fusion_weights/'
    # sbj_name = f'sbj{string_subject[0]}'
    # for n_sbj in range(1, len(string_subject)):
    #     sbj_name += f'.sbj{string_subject[n_sbj]}'
    
    # print(' ---------------------------------------------------------')
    # nowString = datetime.now().strftime('%Y%m%d')
    # if doSave:
    #     print(' - Saving')
    #     np.savez(f'{savePath}{sbj_name}.fusion_weights.{nowString}.npz', weights=weights)


def loadProcess_datasets(subjects, recordingsPath, modelsPath, do_synchronization):
    # Load n number of datasets (n=len(subjects)), apply bci pipeline and if setted the flag, synchronize them

    # Load datasets and process them
    n_subjects = len(subjects)

    probabilities = [None] * n_subjects
    integrated_prob = [None] * n_subjects
    events = [None] * n_subjects
    string_subject = [None] * n_subjects

    for i in range(n_subjects):
        recPath = os.path.join(recordingsPath, subjects[i])
        modpath = os.path.join(modelsPath, subjects[i])
        integrated_prob[i], probabilities[i], events[i], alpha, rejection, string_subject[i] = pipeline_bci(recPath, modpath)

    # # Synchronize probabilities
    # if do_synchronization:
    #     probabilities, events = synchronize_datasets(probabilities, events)

    return integrated_prob, probabilities, events, alpha, rejection, string_subject




def pipeline_bci(recPath, modelPath, alpha=0.96):
    # Load files
    root = tk.Tk()
    selectedFiles = filedialog.askopenfilenames(initialdir=recPath, title='Select MAT files', filetypes=[('MAT files', '*.mat')])
    selectedModel = filedialog.askopenfilename(initialdir=modelPath, title='Select Model file', filetypes=[('Model files', '*.joblib')])
    root.destroy()

    [signal, events_dataFrame, h, fileNames] = get_files(selectedFiles)

    if hasattr(h, 'alpha'):     alpha = h.alpha
    else:                       print(' - Pipeline informations not found. Alpha sets to default (0.96)')
       
    
    modelDictionary = loadmat(os.path.join(modelPath, selectedModel))
    model = modelDictionary['fgmdm']
    fs = modelDictionary['fs']
    laplacian = modelDictionary['laplacian']
    bandPass = modelDictionary['bandPass']
    stopBand = modelDictionary['stopBand']
    filter_order = modelDictionary['filter_order']
    windowsLength = modelDictionary['windowsLength']
    windowsShift = modelDictionary['windowsShift']
    classes = modelDictionary['classes']
    mean_cov = modelDictionary['mean_cov']
    inv_sqrt_mean_cov = modelDictionary['inv_sqrt_mean_cov']



    print(' ---------------------------------------------------------')
    print(' - Data and decoder loaded')

    filt_signal = signal
    if len(bandPass)>0: filt_signal = get_bandranges(filt_signal, bandPass, fs, filter_order, 'bandpass')
    if len(stopBand)>0: filt_signal = get_bandranges(filt_signal, stopBand, fs, filter_order, 'bandstop')

    # if applyLog: filt_signal = utils.get_logpower(filt, fs)  # da sistemare ----------------------------------------------------------------------------------

    lap_signal = filt_signal @ laplacian

    # ## -----------------------------------------------------------------------------    
    wantedChannels = modelDictionary['channels']
    [signal, channels] = select_channels(signal, wantedChannels, actualChannels=channels)

    # ## ----------------------------------------------------------------------------- Covariances
    [covs, cov_events] = get_trNorm_covariance_matrix(lap_signal, events_dataFrame, windowsLength, windowsShift, fs)
    labelVector = get_EventsVector_onFeedback(cov_events, covs.shape[1], classes)
    fdbVector = np.isin(labelVector, classes)

    covs_centered = covs
    if modelDictionary['doRecenter']:
        # Recenter cov matrices, reference = Riemannian mean of training set only 
        mean_cov, _ = get_riemann_mean_covariance(covs[:, fdbVector])
        inv_sqrt_mean_cov = np.expand_dims(invsqrtm(mean_cov), 1)
        print(' - Recentering covariance matrices around eye')
        covs_centered = center_covariances(covs, mean_cov, inv_sqrt_mean_cov)


    probabilities = model.predict_probabilities(covs_centered)
    print(' - Covariance Matrices and classification evaluated')

    # INTEGRATIONS
    probabilities_integrated = probabilities_integration(probabilities, alpha, cov_events)
    print(' - Probabilities integrated')

    return probabilities_integrated, probabilities, cov_events, alpha, #string_subject


def probabilities_integration(data, alpha, events):
    integrated = np.full(data.shape, np.nan)
    integrated[0, :] = do_integration(0.5 * data.shape[1], data[0, :], alpha)
    pos_feedback = events['POS'][events['TYP'] == 781]

    for i in range(1, data.shape[0]):
        if i in pos_feedback:       integrated[i, :] = 0.5 * np.ones(data.shape[1])
        else:                       integrated[i, :] = do_integration(integrated[i-1, :], data[i, :], alpha)
    return integrated

def do_integration(old_data, new_data, alpha):
    if new_data[0] != new_data[1]: new_data = np.array([1, 0]) if new_data[0] > new_data[1] else np.array([0, 1])
    else:   new_data = np.array([0.5, 0.5])
    return old_data * alpha + new_data * (1 - alpha)