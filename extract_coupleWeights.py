import os
import numpy as np
from datetime import datetime
from scipy.io import savemat
import tkinter as tk
from tkinter import filedialog
import numpy as np
from py_utils.signal_processing import get_bandranges, get_trNorm_covariance_matrix, logbandpower
from py_utils.data_managment import get_files, load
from py_utils.eeg_managment import select_channels,get_EventsVector_onFeedback,get_channelsMask
from riemann_utils.covariances import get_riemann_mean_covariance, center_covariances
from pyriemann.utils.base import invsqrtm


def extract_coupleWeights(gammaMI=0, gammaRest=0, doSave=True):

    # Load datasets and process them    
    genPath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    recordingsPath = os.path.join(genPath, 'recordings')
    modelsPath = os.path.join(genPath, 'models')

    _, probabilities, events, _, subjectCodes, classes = loadProcess_datasets(recordingsPath, modelsPath, do_synchronization=True)

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

    weights['loss']['notNormalized']['withRest'] = calc_fusionWeights_crossentropy(probabilities, events, classes, withRest, notLengthNormalization, gamma)
    print(' --- notNormalized.withRest : ')
    print(weights['loss']['notNormalized']['withRest'])
    
    weights['loss']['normalized']['withRest'] = calc_fusionWeights_crossentropy(probabilities, events, classes, withRest, doLengthNormalization, gamma)
    print(' --- normalized.withRest : ')
    print(weights['loss']['normalized']['withRest'])
    
    # Calc weighted average without rest
    weights['loss']['notNormalized']['withoutRest'] = calc_fusionWeights_crossentropy(probabilities, events, classes, noRest, notLengthNormalization, gamma)
    print(' --- notNormalized.withoutRest : ')
    print(weights['loss']['notNormalized']['withoutRest'])
    
    weights['loss']['normalized']['withoutRest'] = calc_fusionWeights_crossentropy(probabilities, events, classes, noRest, doLengthNormalization, gamma)
    print(' --- normalized.withoutRest : ')
    print(weights['loss']['normalized']['withoutRest'])

    # Calc weight with accuracy
    weights['accuracy'] = calc_fusionWeights_accuracy(probabilities, events, classes)
    print('-- ACCURACY :')
    print(weights['accuracy'])

    # Saving
    savePath = os.path.join(genPath, 'weights')
    sbj_name = f'{subjectCodes[0]}'
    for n_sbj in range(1, len(subjectCodes)):
        sbj_name += f'.{subjectCodes[n_sbj]}'
    
    print(' ---------------------------------------------------------')
    nowString = datetime.now().strftime('%Y%m%d')
    if doSave:
        print(' - Saving')
        nameString = f'{sbj_name}.fusion_weights.{nowString}.mat'
        savemat(os.path.join(savePath, nameString), weights)


def loadProcess_datasets(recordingsPath, modelsPath, do_synchronization=True):

    n_subjects = 2

    probabilities = np.empty(n_subjects, dtype=object)
    integrated_prob =  np.empty(n_subjects, dtype=object)
    events = np.empty(n_subjects, dtype=object)
    subjectCodes = np.empty(n_subjects, dtype=object)
    alphas = np.empty(n_subjects)


    for i in range(n_subjects):
        # recPath = os.path.join(recordingsPath, subjects[i])
        # modpath = os.path.join(modelsPath, subjects[i])
        integrated_prob[i], probabilities[i], events[i], alphas[i], filenames = pipeline_bci(recordingsPath, modelsPath)
        subjectCodes[i] = filenames[0].split('/')[-1][:2]

    classes = [769, 770] if 'lhrh' in filenames[0] else [773, 771]

    # Synchronize probabilities
    if do_synchronization:      
        probabilities, _ = synchronize_datasets(probabilities, events)
        integrated_prob, events = synchronize_datasets(integrated_prob, events)

    return integrated_prob, probabilities, events, alphas, subjectCodes, classes




def pipeline_bci(recPath, modelPath, alpha=0.98):
    # Load files
    root = tk.Tk()
    selectedFiles = filedialog.askopenfilenames(initialdir=recPath, title='Select MAT files', filetypes=[('MAT files', '*.mat')])
    selectedModel = filedialog.askopenfilename(initialdir=modelPath, title='Select Model file', filetypes=[('Model files', '*.joblib')], multiple=False)
    root.destroy()

    signal, events_dataFrame, h, filenames = get_files(selectedFiles, ask_user=False)

    if hasattr(h, 'alpha'):     alpha = h.alpha
    else:                       print(' - Pipeline informations not found. Alpha sets to default (0.96)')
       
    modelDictionary = load(selectedModel)
    model = modelDictionary['fgmdm']
    fs = modelDictionary['fs']
    bandPass = modelDictionary['bandPass']
    stopBand = modelDictionary['stopBand']
    filter_order = modelDictionary['filter_order']
    windowsLength = modelDictionary['windowsLength']
    windowsShift = modelDictionary['windowsShift']
    # classes = modelDictionary['classes']
    rejTh = modelDictionary['rejectionThreshold']


    print(' ---------------------------------------------------------')
    print(' - Data and decoder loaded')

    lap_signal = signal @ modelDictionary['laplacian']

    filt_signal = lap_signal
    if len(bandPass)>0: filt_signal = get_bandranges(filt_signal, bandPass, fs, filter_order, 'bandpass')
    if len(stopBand)>0: filt_signal = get_bandranges(filt_signal, stopBand, fs, filter_order, 'bandstop')

    if modelDictionary['applyLog']: filt_signal = logbandpower(filt_signal, fs, slidingWindowLength=modelDictionary['logWindowLength'])


    # ## -----------------------------------------------------------------------------    
    wantedChannels = modelDictionary['channels']
    channels = [ch.replace (" ", "") for ch in h['channels']]
    channelMask = get_channelsMask(wantedChannels, channels)
    filt_signal = filt_signal[:, :, channelMask]

    # ## ----------------------------------------------------------------------------- Covariances
    [covs, cov_events] = get_trNorm_covariance_matrix(filt_signal, events_dataFrame, windowsLength, windowsShift, fs)
    # labelVector = get_EventsVector_onFeedback(cov_events, covs.shape[1], classes)

    covs_centered = covs
    if modelDictionary['mean_cov'] is not None:
        # Recenter cov matrices, reference = Riemannian mean of training set only 
        print(' - Recentering covariance matrices around eye')
        covs_centered = center_covariances(covs, modelDictionary['mean_cov'], modelDictionary['inv_sqrt_mean_cov'])

    probabilities = model.predict_probabilities(covs_centered)
    print(' - Covariance Matrices and classification evaluated')
    # INTEGRATIONS
    probabilities_integrated = probabilities_integration(probabilities, alpha, cov_events, rejTh)
    print(' - Probabilities integrated')
    return probabilities_integrated, probabilities, cov_events, alpha, filenames



def probabilities_integration(data, alpha, events, rejTh=0.5):
    if data.ndim == 2:
        data = np.expand_dims(data, axis=0)

    integrated = np.full(data.shape, np.nan)
    integrated[:, 0] = data[:, 0]
    pos_feedback = events['pos'][events['typ'] == 781]

    for bId in range(data.shape[0]):
        for i in range(1, data.shape[1]):
            if i in pos_feedback:       integrated[bId, i] = 0.5 * np.ones(data.shape[2])
            else:                       integrated[bId, i] = do_integration(integrated[bId, i-1], data[bId, i], alpha, rejTh)
    return integrated


def do_integration(old_data, new_data, alpha, rejTh=0.5):
    if np.max(new_data) < rejTh:    return old_data
    if new_data[0] != new_data[1]: new_data = np.array([1, 0]) if new_data[0] > new_data[1] else np.array([0, 1])
    else:   new_data = np.array([0.5, 0.5])
    return old_data * alpha + new_data * (1 - alpha)


def synchronize_datasets(probabilities, events):
    i = 1
    flag = np.zeros((3, len(probabilities) - 1), dtype=int)
    
    while True:
        tpos = events[0]['pos']
        tdur = events[0]['dur']
        ttyp = events[0]['typ']

        if (tdur != events[i]['dur']).any():    raise ValueError('EVENTS ARE DIFFERENT')
        else:                           flag[0, i - 1] = 1

        if (ttyp != events[i]['typ']).any():    raise ValueError('EVENTS ARE DIFFERENT')
        else:                                       flag[1, i - 1] = 1

        # check if the pos are just shifted equally, if so move them and the probabilities
        if not np.array_equal(tpos, events[i]['pos']):
            offset = tpos[0] - events[i]['pos'][0]

            if not np.array_equal(tpos, events[i]['pos']) and np.all(tpos - events[i]['pos'] == offset):
                if offset > 0:
                    events[0]['pos'] = events[0]['pos'] - offset
                    probabilities[0] = probabilities[0][offset:]
                    probabilities[i] = probabilities[i][:-offset]
                else:
                    events[i]['pos'] = events[i]['pos'] - offset
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
        

def calc_fusionWeights_crossentropy(probabilities, events, classes, withRest=True, doLengthNormalization=True, gamma=None):
    
    if gamma is not None and len(gamma) == 2:
        gammaMI = gamma[0]
        gammaRest = gamma[1]
    elif gamma is not None:
        gammaMI = gamma[0]
        gammaRest = 0
    else:
        gammaMI = 0
        gammaRest = 0


    weights = np.nan * np.ones(len(probabilities))

    for n_sbj in range(len(probabilities)):
        evnts = events[n_sbj] if isinstance(events, list) else events
        mi_vector, rest_vector, mi_trials_vector, rest_trials_vector = create_classes_vectors(evnts, classes, probabilities[n_sbj].shape[1])

        loss = binary_crossEntropy(probabilities[n_sbj], mi_vector, mi_trials_vector, gammaMI, doLengthNormalization)
        if withRest:
            loss += crossEntropy_rest(probabilities[n_sbj], rest_vector, rest_trials_vector, gammaRest, doLengthNormalization)
        weights[n_sbj] = 1 / loss

    return weights



def create_classes_vectors(events, classes, n_points):
    mi_vector = np.nan * np.ones(n_points)
    rest_vector = np.nan * np.ones(n_points)

    mi_trials_vector = np.nan * np.ones(n_points)
    rest_trials_vector = np.nan * np.ones(n_points)

    mi_trials = 1
    rest_trials = 1

    for idx in np.where(events['typ'] == 781)[0]:

        column = np.where(classes == events['typ'][idx - 1])[0]
        duration = np.arange(events['dur'][idx])

        if column.size == 0:  # it means it's rest
            rest_vector[events['pos'][idx] + duration] = 1
            rest_trials_vector[events['pos'][idx] + duration] = rest_trials
            rest_trials += 1
        else:
            mi_vector[events['pos'][idx] + duration] = column[0]
            mi_trials_vector[events['pos'][idx] + duration] = mi_trials
            mi_trials += 1

    return mi_vector, rest_vector, mi_trials_vector, rest_trials_vector


def binary_crossEntropy(probabilities, classes_vector, trials_vector, gamma=0, do_length_normalization=False):
    # classes vector of length size(probabilities,1) where is 0 or 1 depending
    # on the class of that data point, nan otherwise

    idx_nan = np.isnan(classes_vector)
    classesK = classes_vector[~idx_nan]
    probability = probabilities[:,~idx_nan,0]   # i need only the last column due to the fact that p1 = 1-p0
    trialsK = trials_vector[~idx_nan]

    binary_loss = np.array([0.0, 0.0])
    binary_N = np.array([0, 0])

    n_max_trials = int(np.max(trialsK))

    for n_trial in range(1, n_max_trials + 1):
        idx_trial = np.where(trialsK == n_trial)[0]

        class_label = classesK[idx_trial[0]]
        idx_loss = int(class_label)

        t_binary_loss = 0.0

        for idx in idx_trial:
            probab = np.squeeze(probability[:,idx])

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

    idx_nan = np.isnan(classes_vector)
    probability = probabilities[:,~idx_nan]
    trialsK = trials_vector[~idx_nan]

    if trialsK.size == 0:
        print(' -- No rest trials found, skipping rest loss calculation')
        return 0

    focus = lambda x: np.abs(0.5 - x) ** gamma
    map_probab = lambda x: -2 * np.abs(0.5 - x) + 1

    n_max_trials = int(np.max(trialsK))
    rest_loss = 0

    for n_trial in range(1, n_max_trials + 1):
        idx_trial = np.where(trialsK == n_trial)[0]

        t_rest_loss = 0

        for idx in idx_trial:
            probab = np.squeeze(probability[:,idx])
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
        
        result = create_classes_vectors(evnts, column_classes, probabilities[n_sbj].shape[1])
        mi_vector[n_sbj] = result[0]
        mi_trials_vector[n_sbj] = result[2]
    
    weights = np.full(len(probabilities), np.nan)
    
    for n_sbj in range(len(probabilities)):
        accuracy = points_accuracy(probabilities[n_sbj], mi_vector[n_sbj], mi_trials_vector[n_sbj])
        weights[n_sbj] = accuracy
    
    return weights

def points_accuracy(probabilities, classes_vector, trials_vector):
    
    idx_nan = np.isnan(classes_vector)
    classesK = classes_vector[~idx_nan]
    probability = probabilities[:, ~idx_nan]
    trialsK = trials_vector[~idx_nan]

    accuracy = 0


    n_max_trials = int(np.max(trialsK))

    for n_trial in range(1, n_max_trials + 1):

        idx_trial = np.where(trialsK == n_trial)[0]
        class_val = classesK[idx_trial[0]]

        prob = np.squeeze(probability[:, idx_trial])
        max_column = np.argmax(prob, axis=1)

        n_corrected_points = np.sum(max_column == class_val + 1)
        accuracy = accuracy + n_corrected_points / len(idx_trial)                             

    accuracy = accuracy / n_max_trials
    
    return accuracy


