import sys, os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


import numpy as np
from datetime import datetime
from scipy.io import savemat
import tkinter as tk
from tkinter import filedialog
import numpy as np
from py_utils.signal_processing import get_bandranges, get_trNorm_covariance_matrix, logbandpower, get_covariance_matrix_normalized
from py_utils.data_managment import get_files, load
from py_utils.eeg_managment import select_channels,get_EventsVector_onFeedback,get_channelsMask
from riemann_utils.covariances import get_riemann_mean_covariance, center_covariances
from pyriemann.utils.base import invsqrtm
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from scipy.special import softmax


def extract_coupleWeights(n_subjects=2, doSave=True):

    # Load datasets and process them    
    genPath = os.path.join(os.path.abspath(os.path.join(current_dir, '..')), 'data')

    recordingsPath = os.path.join(genPath, 'recordings')
    modelsPath = os.path.join(genPath, 'models')

    model, covs, cov_events, _, filenames = covProcesPipeline_datasets(recordingsPath, modelsPath, n_subjects=n_subjects, cutStartEnd=False)

    f = np.empty(n_subjects, dtype=object)
    s = np.empty(n_subjects, dtype=object)
    for nSbj in range(n_subjects):
        probabilities = np.squeeze(model[nSbj].predict_probabilities(covs[nSbj]))
        y = np.full(probabilities.shape[0], np.nan)
        ev = cov_events[nSbj]
        idx = ev[ev['typ'] == 781].index
        classes = np.unique(ev['typ'][idx-1])
        for i in idx:
            pos = ev['pos'][i]
            dur = ev['dur'][i]
            cue = ev['typ'][i - 1]
            y[pos:pos+dur] = cue == classes[1]  # binary labels

        f[nSbj] = compute_model_features(probabilities[~np.isnan(y),1], y[~np.isnan(y)].astype(int))
        s[nSbj] = compute_model_score(f[nSbj])

    weights = normalize_scores_to_weights([s[0], s[1]], beta=4.0)
    print("Model weights:", weights)

    # Saving
    subjectCodes = np.empty(n_subjects, dtype=object)
    for i in range(n_subjects):
        subjectCodes[i] = filenames[i][0].split('/')[-1][:2]
    savePath = os.path.join(genPath, 'weights')
    sbj_name = f'{subjectCodes[0]}'
    for n_sbj in range(1, len(subjectCodes)):
        sbj_name += f'.{subjectCodes[n_sbj]}'
    
    print(' ---------------------------------------------------------')
    nowString = datetime.now().strftime('%Y%m%d')
    if doSave:
        print(' - Saving')
        nameString = f'{sbj_name}.fusion_weights.{nowString}.mat'
        savemat(os.path.join(savePath, nameString), {'weights': weights})



def compute_model_features(p, y, other_p=None, n_bins=10):
    """
    Compute informative features from predicted probabilities and true labels.

    Parameters
    ----------
    p : array (N,)  -- predicted prob of class 1
    y : array (N,)  -- true labels {0,1}
    other_p : array (N,), optional -- other model's probs (for agreement features)
    n_bins : int -- bins for ECE calculation

    Returns
    -------
    dict of feature_name -> value
    """
    eps = 1e-12

    p = np.clip(p, eps, 1 - eps)
    y = np.asarray(y)
    N = len(y)

    # Core statistics
    mean_conf = p.mean()
    prob_true = p * (y == 1) + (1 - p) * (y == 0)
    mean_conf_true = prob_true.mean()

    nll = log_loss(y, p)
    brier = brier_score_loss(y, p)

    ent = -(p * np.log(p) + (1 - p) * np.log(1 - p)).mean()
    margin = (np.abs(p - 0.5) * 2).mean()

    try:
        auc = roc_auc_score(y, p)
    except Exception:
        auc = 0.5

    # --- Expected Calibration Error (ECE) ---
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    p_bar = y.mean()
    resolution = 0.0
    bin_idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    for b in range(n_bins):
        mask = (bin_idx == b)
        if not np.any(mask):
            continue
        pb = p[mask].mean()
        yb = y[mask].mean()
        wb = mask.sum() / N
        ece += wb * abs(pb - yb)
        resolution += wb * (yb - p_bar) ** 2

    # Optional cross-model agreement
    if other_p is not None:
        other_p = np.clip(other_p, eps, 1 - eps)
        kl = (p * np.log(p / other_p) + (1 - p) * np.log((1 - p) / (1 - other_p))).mean()
        l1 = np.abs(p - other_p).mean()
    else:
        kl = np.nan
        l1 = np.nan

    return {
        "mean_conf": mean_conf,
        "mean_conf_true": mean_conf_true,
        "nll": nll,
        "brier": brier,
        "entropy": ent,
        "margin": margin,
        "auc": auc,
        "ece": ece,
        "resolution": resolution,
        "kl_with_other": kl,
        "l1_with_other": l1
    }


def compute_model_score(features, feature_weights=None):
    """
    Convert feature dict into a single numeric score (higher = better).
    Flip signs automatically for metrics where lower is better.
    """
    default_weights = {
        "mean_conf_true": 1.0,
        "nll": 1.0,
        "brier": 1.0,
        "ece": 0.8,
        "resolution": 0.8,
        "auc": 1.0,
        "entropy": 0.6,
        "margin": 0.7,
        "l1_with_other": -0.2
    }
    if feature_weights is None:
        feature_weights = default_weights

    lower_is_better = {"nll", "brier", "ece", "entropy", "kl_with_other"}
    score = 0.0
    for k, w in feature_weights.items():
        if k not in features or features[k] is None or np.isnan(features[k]):
            continue
        v = features[k]
        if k in lower_is_better:
            v = -v
        score += w * v
    return score


def normalize_scores_to_weights(scores, beta=3.0):
    """
    Convert a list of model scores into normalized weights using softmax.
    """
    w = softmax(beta * np.array(scores))
    return w / w.sum()








def covProcesPipeline_datasets(recPath, modelPath, n_subjects=2, cutStartEnd=False):

    model = np.empty(n_subjects, dtype=object)
    covs_centered =  np.empty(n_subjects, dtype=object)
    cov_events = np.empty(n_subjects, dtype=object)
    rejTh = np.empty(n_subjects)
    filenames = np.empty(n_subjects, dtype=object)

    for i in range(n_subjects):

        root = tk.Tk()
        selectedFiles = filedialog.askopenfilenames(initialdir=recPath, title='Select MAT files', filetypes=[('MAT files', '*.mat')])
        selectedModel = filedialog.askopenfilename(initialdir=modelPath, title='Select Model file', filetypes=[('Model files', '*.joblib')], multiple=False)
        root.destroy()

        signal, events_dataFrame, h, filenames[i] = get_files(selectedFiles, ask_user=False, cutStartEnd=cutStartEnd)

        modelDictionary = load(selectedModel)
        model[i] = modelDictionary['fgmdm']
        fs = modelDictionary['fs']
        bandPass = modelDictionary['bandPass']
        stopBand = modelDictionary['stopBand']
        filter_order = modelDictionary['filter_order']
        windowsLength = modelDictionary['windowsLength']
        windowsShift = modelDictionary['windowsShift']
        # classes = modelDictionary['classes']
        rejTh[i] = modelDictionary['rejectionThreshold']


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

        [covs, cov_events[i]] = get_covariance_matrix_normalized(filt_signal, events_dataFrame, windowsLength, windowsShift, fs, normalizationMethod=modelDictionary['normalizationMethod'])

    
        if modelDictionary['mean_cov'] is not None:
            # Recenter cov matrices, reference = Riemannian mean of training set only 
            print(' - Recentering covariance matrices around eye')
            covs_centered[i] = center_covariances(covs, modelDictionary['mean_cov'], modelDictionary['inv_sqrt_mean_cov'])
        else:
            covs_centered[i] = covs

    return model, covs_centered, cov_events, rejTh, filenames











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
        integrated_prob[i], probabilities[i], events[i], alphas[i], filenames = pipeline_bci(recordingsPath, modelsPath, cutStartEnd=False)
        subjectCodes[i] = filenames[0].split('/')[-1][:2]

    classes = [769, 770] if 'lhrh' in filenames[0] else [773, 771]

    # Synchronize probabilities
    if do_synchronization:      
        probabilities, _ = synchronize_datasets(probabilities, events)
        integrated_prob, events = synchronize_datasets(integrated_prob, events)

    return integrated_prob, probabilities, events, alphas, subjectCodes, classes



def pipeline_bci(recPath, modelPath, cutStartEnd=False, alpha=0.99):
    # Load files
    model, covs_centered, cov_events, alpha, rejTh, filenames = covProcesPipeline_datasets(recPath, modelPath, cutStartEnd=cutStartEnd)
    

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
    if np.max(new_data) < rejTh:    new_data = np.array([0.5, 0.5])
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

        diff = tpos.values - events[i]['pos'].values
        # check if the pos are just shifted equally, if so move them and the probabilities
        if np.any(abs(diff)>1):
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
            elif len(np.unique(tpos - events[i]['pos']))>1:
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


