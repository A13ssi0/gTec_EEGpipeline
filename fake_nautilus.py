import numpy as np
from functions import proc_spectrogram
from scipy.io import loadmat, savemat
import time
from py_utils.eeg_managment import select_channels
from py_utils.signal_processing import get_bandranges, get_covariance_matrix_traceNorm_online
from pyriemann.utils.test import is_sym_pos_def
from buffer import Buffer

     
def main(array_size):
    data = loadmat('data_test.mat')
    data = data['a']
    start = 0
    buffer_shape = (500, data.shape[1])
    buffer = Buffer(buffer_shape)
    classifier_dict = None  # Placeholder for classifier, to be defined later

    while start < data.shape[0]:
        end = start + array_size
        if end > data.shape[0]:     return
        process_data(data[start:end], buffer, classifier_dict)
        start += array_size


    
def process_data(data, buffer, classifier_dict):
#     wlength =    0.5
#     wshift =        0.0625
#     mlength =       1
#    # wconv =         'backward';
#     pshift =        0.25
#     fs = 512
#     a = time.time()
#     print(data.shape)
#     psd,f = proc_spectrogram(data, wlength, wshift, pshift, fs, mlength)
#     print(time.time()-a)
#     #savemat('psd_test.mat', {'psd': psd, 'f': f})
    
    buffer.add_data(data)
    if buffer.isFull:
        # apply laplacian
        do_classification(buffer.data , classifier_dict)
        pass
        

def do_classification(eeg, classifier_dict):

    # classifier_dict = args[0]
    # n_channels = args[1]
    # classifier_type = args[2]
    # pub = args[3]
    
    classifier = classifier_dict['fgmdm']

    [eeg, channels] = select_channels(eeg, classifier_dict['wantedChannels'])
    #eeg = utils.apply_ROI_over_channels(eeg, channels, classifier_dict['channelGroups'])
    #eeg_unfilt = np.expand_dims(eeg, axis=0)
    # eeg = np.expand_dims(eeg, axis=0)
    eeg = get_bandranges(eeg, classifier_dict['bandranges'], classifier_dict['fs'], classifier_dict['filter_order'], classifier_dict['stopranges'])
    #print(eeg.shape)
    cov = get_covariance_matrix_traceNorm_online(eeg)
    #print(cov.shape)
    # center cov matrices wrt ref matrix used for the classifier creation
    # cov_centered = utils.center_covariance_online(cov, classifier_dict['inv_sqrt_mean_cov'])
    cov_centered = cov
    
    # if data_cov.shape[1]==512:
    #     data_covs = np.append(data_covs, data_cov, axis=0)
    #     all_covs = np.append(all_covs, cov_centered, axis=0)
    #     #all_unfilt =  np.append(all_unfilt, eeg_unfilt, axis=0)

    if not is_sym_pos_def(cov_centered):
        # pass
        print('not SPD')
        val, _ = np.linalg.eig(np.squeeze(cov_centered))
        print(val)
        pred_proba = np.empty((1,1,2))
        pred_proba[0,0] = np.array([0.5, 0.5])
    else:
        pred_proba = classifier.predict_probabilities(cov_centered)
     
    # # pred = classifier.merge_classifiers(pred_proba)[0]
    # # EDITED
    # # pred = classifier.merge_bands(pred_proba)[0]
    # #pred = pred/np.sum(pred)
    # prediction.softpredict.data = pred_proba.flatten()
    # prediction.header.stamp = rospy.Time.now()
    # prediction.decoder.classes = classifier_dict['classes']

    # # if all_covs.shape[0]==600 :
    # #     scipy.io.savemat('/home/mtld/mtld_workspace/'+ subject+'_'+today+'_ros_covs.mat', {'ros_covs':all_covs, 'ros_data': data_covs})#, 'data_unfilt': all_unfilt})


    # pub.publish(prediction)
    

main(25)  