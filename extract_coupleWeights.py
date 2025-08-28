import os
import numpy as np
from datetime import datetime
from scipy.io import loadmat
import tkinter as tk
from tkinter import filedialog
import numpy as np
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

    # Variables for calc weights
    doLengthNormalization = True
    notLengthNormalization = False
    withRest = True
    noRest = False

    gamma = [gammaMI, gammaRest]
    
    # Calc weighted average with rest
    print(' ---------------------------------------------------------')
    print('- WEIGHTS -')

    print('-- LOSS --')
    weights = {'loss': {}}
    weights['loss']['notNormalized'] = {}
    weights['loss']['normalized'] = {}

    weights['loss']['notNormalized']['withRest'] = calc_fusionWeights_crossentropy(probabilities, events, withRest, notLengthNormalization, classes, gamma)
    print(' --- notNormalized.withRest : ')
    print(weights['loss']['notNormalized']['withRest'])
    
    weights['loss']['normalized']['withRest'] = calc_fusionWeights_crossentropy(probabilities, events, withRest, doLengthNormalization, classes, gamma)
    print(' --- normalized.withRest : ')
    print(weights['loss']['normalized']['withRest'])
    
    # Calc weighted average without rest
    weights['loss']['notNormalized']['withoutRest'] = calc_fusionWeights_crossentropy(probabilities, events, noRest, notLengthNormalization, classes, gamma)
    print(' --- notNormalized.withoutRest : ')
    print(weights['loss']['notNormalized']['withoutRest'])
    
    weights['loss']['normalized']['withoutRest'] = calc_fusionWeights_crossentropy(probabilities, events, noRest, doLengthNormalization, classes, gamma)
    print(' --- normalized.withoutRest : ')
    print(weights['loss']['normalized']['withoutRest'])

    # Calc weight with accuracy
    weights['accuracy'] = calc_fusionWeights_accuracy(probabilities, events, classes)
    print('-- ACCURACY :')
    print(weights['accuracy'])

    # Saving
    savePath = os.path.join(genPath, 'weights')
    sbj_name = f'sbj{string_subject[0]}'
    for n_sbj in range(1, len(string_subject)):
        sbj_name += f'.sbj{string_subject[n_sbj]}'
    
    print(' ---------------------------------------------------------')
    nowString = datetime.now().strftime('%Y%m%d')
    if doSave:
        print(' - Saving')
        np.savez(f'{savePath}{sbj_name}.fusion_weights.{nowString}.npz', weights=weights)


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

    # Synchronize probabilities
    if do_synchronization:
        probabilities, events = synchronize_datasets(probabilities, events)

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

    return probabilities_integrated, probabilities, cov_events, alpha, string_subject


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


def synchronize_datasets(probabilities, events):
    i = 1
    flag = np.zeros((3, len(probabilities) - 1), dtype=int)
    
    while True:
        tpos = events[0]['POS']
        tdur = events[0]['DUR']
        ttyp = events[0]['TYP']

        if tdur != events[i]['DUR']:    raise ValueError('EVENTS ARE DIFFERENT')
        else:                           flag[0, i - 1] = 1

        if ttyp != events[i]['TYP']:    raise ValueError('EVENTS ARE DIFFERENT')
        else:                           flag[1, i - 1] = 1

        # check if the pos are just shifted equally, if so move them and the probabilities
        if not np.array_equal(tpos, events[i]['POS']):
            offset = tpos[0] - events[i]['POS'][0]

            if not np.array_equal(tpos, events[i]['POS']) and np.all(tpos - events[i]['POS'] == offset):
                if offset > 0:
                    events[0]['POS'] = events[0]['POS'] - offset
                    probabilities[0] = probabilities[0][offset:]
                    probabilities[i] = probabilities[i][:-offset]
                else:
                    events[i]['POS'] = events[i]['POS'] - offset
                    probabilities[0] = probabilities[0][:-offset]
                    probabilities[i] = probabilities[i][offset:]
            else:
                raise ValueError('EVENTS ARE DIFFERENT')
        else:
            flag[2, i - 1] = 1

        if i == len(probabilities) - 1:
            i = 1
        else:
            i += 1

        if np.all(np.all(flag)):
            events = events[0]
            print(' - Probabilities synchronized')
            return probabilities, events
        

def calc_fusionWeights_crossentropy(probabilities, events, withRest=True, doLengthNormalization=True, column_classes=None, gamma=None):
    
    if gamma is not None and len(gamma) == 2:
        gammaMI = gamma[0]
        gammaRest = gamma[1]
    elif gamma is not None:
        gammaMI = gamma[0]
        gammaRest = 0
    else:
        gammaMI = 0
        gammaRest = 0

    if column_classes is None:
        column_classes = [773, 771]

    weights = np.nan * np.ones(len(probabilities))

    for n_sbj in range(len(probabilities)):
        evnts = events[n_sbj] if isinstance(events, list) else events
        mi_vector, rest_vector, mi_trials_vector, rest_trials_vector = create_classes_vectors(evnts, column_classes, probabilities[n_sbj].shape[0])

        loss = binary_crossEntropy(probabilities[n_sbj], mi_vector, mi_trials_vector, gammaMI, doLengthNormalization)
        if withRest:
            loss += crossEntropy_rest(probabilities[n_sbj], rest_vector, rest_trials_vector, gammaRest, doLengthNormalization)
        weights[n_sbj] = 1 / loss

    return weights



def create_classes_vectors(events, column_classes, n_points):
    mi_vector = np.nan * np.ones(n_points)
    rest_vector = np.nan * np.ones(n_points)

    mi_trials_vector = np.nan * np.ones(n_points)
    rest_trials_vector = np.nan * np.ones(n_points)

    mi_trials = 1
    rest_trials = 1

    for idx in np.where(events['TYP'] == 781)[0]:

        column = np.where(column_classes == events['TYP'][idx - 1])[0]
        duration = np.arange(events['DUR'][idx])

        if column.size == 0:  # it means it's rest
            rest_vector[events['POS'][idx] + duration] = 1
            rest_trials_vector[events['POS'][idx] + duration] = rest_trials
            rest_trials += 1
        else:
            mi_vector[events['POS'][idx] + duration] = column[0] - 1
            mi_trials_vector[events['POS'][idx] + duration] = mi_trials
            mi_trials += 1

    return mi_vector, rest_vector, mi_trials_vector, rest_trials_vector


def binary_crossEntropy(probabilities, classes_vector, trials_vector, gamma=0, do_length_normalization=False):
    # classes vector of length size(probabilities,1) where is 0 or 1 depending
    # on the class of that data point, nan otherwise

    if probabilities.ndim > 1:
        probabilities = probabilities[:, 0]

    idx_nan = np.isnan(classes_vector)
    classes_vector = classes_vector[~idx_nan]
    probabilities = probabilities[~idx_nan]
    trials_vector = trials_vector[~idx_nan]

    binary_loss = np.array([0.0, 0.0])
    binary_N = np.array([0, 0])

    n_max_trials = int(np.max(trials_vector))

    for n_trial in range(1, n_max_trials + 1):
        idx_trial = np.where(trials_vector == n_trial)[0]

        class_label = classes_vector[idx_trial[0]]
        idx_loss = int(class_label)

        t_binary_loss = 0.0

        for idx in idx_trial:
            probab = probabilities[idx]

            t_binary_loss1 = class_label * (1 - probab) ** gamma * np.log(probab)
            t_binary_loss0 = (1 - class_label) * probab ** gamma * np.log(1 - probab)

            t_binary_loss += t_binary_loss1 + t_binary_loss0

        if do_length_normalization:
            length_trial = len(idx_trial)
        else:
            length_trial = 1

        binary_loss[idx_loss] += t_binary_loss / length_trial
        binary_N[idx_loss] += 1

    alpha = -len(classes_vector) / binary_N
    binary_loss = alpha * binary_loss

    loss = np.sum(binary_loss)
    return loss



def crossEntropy_rest(probabilities, classes_vector, trials_vector, gamma=0, do_length_normalization=False):
    if probabilities.shape[1] > 1:
        probabilities = probabilities[:, 0]

    idx_nan = np.isnan(classes_vector)
    probability = probabilities[~idx_nan]
    trials_vector = trials_vector[~idx_nan]

    focus = lambda x: np.abs(0.5 - x) ** gamma
    map_probab = lambda x: -2 * np.abs(0.5 - x) + 1

    n_max_trials = int(np.max(trials_vector))
    rest_loss = 0

    for n_trial in range(1, n_max_trials + 1):
        idx_trial = np.where(trials_vector == n_trial)[0]

        t_rest_loss = 0

        for idx in idx_trial:
            probab = probability[idx]
            t_rest_loss += focus(probab) * np.log(map_probab(probab))

        if do_length_normalization:
            length_trial = len(idx_trial)
        else:
            length_trial = 1

        rest_loss += t_rest_loss / length_trial

    rest_N = len(probability)
    alpha = -1 / rest_N
    rest_loss = alpha * rest_loss

    loss = np.sum(rest_loss)
    return loss



def calc_fusionWeights_accuracy(probabilities, events, column_classes=None):
    
    if column_classes is None:
        column_classes = [773, 771]
    
    mi_vector = {}
    mi_trials_vector = {}
    
    for n_sbj in range(len(probabilities)):
        if isinstance(events, (list, tuple)) and len(events) > n_sbj:
            evnts = events[n_sbj]
        else:
            evnts = events
        
        result = create_classes_vectors(evnts, column_classes, probabilities[n_sbj].shape[0])
        mi_vector[n_sbj] = result[0]
        mi_trials_vector[n_sbj] = result[2]
    
    weights = np.full(len(probabilities), np.nan)
    
    for n_sbj in range(len(probabilities)):
        accuracy = points_accuracy(probabilities[n_sbj], mi_vector[n_sbj], mi_trials_vector[n_sbj])
        weights[n_sbj] = accuracy
    
    return weights

import numpy as np

def points_accuracy(probabilities, classes_vector, trials_vector):
    
    idx_nan = np.isnan(classes_vector)
    classes_vector = classes_vector[~idx_nan]
    probabilities = probabilities[~idx_nan, :]
    trials_vector = trials_vector[~idx_nan]

    accuracy = 0


    n_max_trials = int(np.max(trials_vector))

    for n_trial in range(1, n_max_trials + 1):

        idx_trial = np.where(trials_vector == n_trial)[0]
        class_val = classes_vector[idx_trial[0]]

        prob = probabilities[idx_trial, :]
        max_column = np.argmax(prob, axis=1)

        n_corrected_points = np.sum(max_column == class_val + 1)
        accuracy = accuracy + n_corrected_points / len(idx_trial)                             

    accuracy = accuracy / n_max_trials
    
    return accuracy


