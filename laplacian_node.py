#!/usr/bin/env python3

import rospy
from scipy.io import loadmat
from rosneuro_msgs.msg  import NeuroFrame
import numpy as np
from rospy.numpy_msg import numpy_msg
from std_msgs.msg import Float64MultiArray

def apply_laplacian(data, args):
    laplacian = args[0]
    publisher = args[1]
    # EDITED
    #eeg = np.reshape(data.eeg.data, (data.eeg.info.nchannels, data.eeg.info.nsamples)).T
    eeg = np.reshape(data.eeg.data, (data.eeg.info.nsamples, data.eeg.info.nchannels))
    eeg_lap = eeg @ laplacian   
    msg = Float64MultiArray()
    msg.data = eeg_lap.flatten()
    publisher.publish(msg)
     

if __name__ == '__main__':
    rospy.init_node('laplacian_applier', anonymous=False)

    # Load Laplacian matrix
    path_classifier = rospy.get_param('~lap_path')
    laplacian = loadmat(path_classifier)
    lapmask = laplacian['lapmask']

    publisher = rospy.Publisher('neurodata_lap', Float64MultiArray, queue_size=1)
    # EDITED
    rospy.Subscriber('neurodata_filtered', NeuroFrame, apply_laplacian, (lapmask, publisher), queue_size=1)

    rospy.spin()