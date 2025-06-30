from matplotlib import mlab
from scipy import signal
import warnings
import numpy as np

def proc_spectrogram(data, wlength, wshift, pshift, samplerate, mlength=None):
    """
    [features, f] = proc_spectrogram(data, wlength, wshift, pshift, samplerate [, mlength])
    
    The function computes the spectrogram on the real data.
    
    Input arguments:
        - data              Data matrix [samples x channels]
        - wlength           Window's lenght to be used to segment data and
                           compute the spectrogram                             [in seconds]
        - wshift            Shift of the external window (e.g., frame size)     [in seconds]
        - pshift            Shift of the internal psd windows                   [in seconds]
        - samplerate        Samplerate of the data
        - [mlength]         Optional length of the external windows to compute
                           the moving average.                                 [in seconds] 
                           By default the length of the moving average window
                           is set to 1 second. To not compute the moving
                           average, empty argument can be provided.
    
    Output arguments:
        - features          Output of the spectrogram in the format: 
                           [windows x frequencies x channels]. Number of
                           windows (segments) is computed according to the
                           following formula: 
                           nsegments = fix((NX-NOVERLAP)/(length(WINDOW)-NOVERLAP))
                           where NX is the total number of samples, NOVERLAP
                           the number of overlapping samples for each segment
                           and length(WINDOW) the number of samples in each
                           segment. 
                           Number of frequencies is computed according to the
                           NFFT. nfrequencies is equal to (NFFT/2+1) if NFFT 
                           is even, and (NFFT+1)/2 if NFFT is odd. NFFT is the
                           maximum between 256 and the next power of 2 greater
                           than the length(WINDOW).
        - f                 Vectore with the computed frequencies
    
    SEE ALSO: spectrogram, nextpow2
    """
    
    # Data informations
    nsamples = data.shape[0]
    nchannels = data.shape[1]

    # Useful params for PSD extraction with the fast algorithm
    psdshift = pshift * samplerate
    winshift = wshift * samplerate

    if (psdshift % winshift != 0) and (winshift % psdshift != 0):
        warnings.warn('[proc_spectrogram] The fast PSD method cannot be applied with the current settings!', 
                     category=UserWarning)
        raise ValueError('[proc_spectrogram] The internal welch window shift must be a multiple of the overall feature window shift (or vice versa)!')

    # Create arguments for spectrogram
    spec_win = int(wlength * samplerate)
    
    # Careful here: The overlapping depends on whether the winshift or the
    # psdshift is smaller. Some calculated internal windows will be redundant,
    # but the speed is much faster anyway

    if psdshift <= winshift:
        spec_ovl = spec_win - int(psdshift)
    else:
        spec_ovl = spec_win - int(winshift)

    # Calculate all the internal PSD windows
    nsegments = int((nsamples - spec_ovl) / (spec_win - spec_ovl))  # From spectrogram's help page
    nfft = max(256, int(2**(np.ceil(np.log2(spec_win)))))  # nextpow2 equivalent
    if nfft % 2 == 0:
        nfreqs = (nfft // 2) + 1
    else:
        nfreqs = (nfft + 1) // 2
    
    psd = np.zeros((nfreqs, nsegments, nchannels))
    window = np.hamming(samplerate* wlength)  # Hamming window for the spectrogram
    #NOverlap = wlength * samplerate  # Overlap for the spectrogram

    for chId in range(nchannels):
        #f, t, Sxx = signal.spectrogram(data[:, chId], fs=samplerate, window='hamming', 
        #                              nperseg=spec_win, noverlap=spec_ovl, nfft=nfft)
        #[~,f,~,psd(:,:,chId)] = spectrogram(data(:,chId), spec_win, spec_ovl, [], samplerate)
       [psd[:, :, chId], f, t] = mlab.specgram(data[:, chId], NFFT = nfft, Fs = samplerate, window = window, noverlap = spec_ovl)
        #psd[:, :, chId] = Sxx
    
    if mlength is not None:
        # Setup moving average filter parameters
        mavg_a = 1
        if winshift >= psdshift:
            # Case where internal windows are shifted according to psdshift
            mavgsize = int(((mlength * samplerate) / psdshift) - 1)
            mavg_b = (1 / mavgsize) * np.ones(mavgsize)
            mavg_step = int(winshift / psdshift)
        else:
            # Case where internal windows are shifted according to winshift
            mavgsize = int(((mlength * samplerate) / winshift) - (psdshift / winshift))
            mavg_b = np.zeros(mavgsize)
            step_size = int(psdshift / winshift)
            mavg_b[0:mavgsize-1:step_size] = 1
            mavg_b = mavg_b / np.sum(mavg_b)
            mavg_step = 1
        
        # Find last non-zero element (equivalent to find(mavg_b~=0, 1, 'last'))
        startindex = np.where(mavg_b != 0)[0][-1]

        # Apply filter along axis 1 (equivalent to filter(mavg_b,mavg_a,psd,[],2))
        features = signal.lfilter(mavg_b, mavg_a, psd, axis=1)
        # Permute dimensions: [2 1 3] -> transpose from (nfreqs, nsegments, nchannels) to (nsegments, nfreqs, nchannels)
        features = np.transpose(features, (1, 0, 2))

        # Get rid of initial filter byproducts
        features = features[startindex:, :, :]

        # In case of psdshift, there will be redundant windows. Remove them
        if mavg_step > 1:
            features = features[::mavg_step, :, :]
    else:
        features = psd
        # Permute dimensions: [2 1 3] -> transpose from (nfreqs, nsegments, nchannels) to (nsegments, nfreqs, nchannels)
        features = np.transpose(features, (1, 0, 2))
    
    return features, f


# import numpy as np
# from scipy.fft import rfft, rfftfreq

# class Pwelch:

#     def __init__(self, wlength, wshift, pshift, samplerate, mlength=None):
#         self.config.wlength = wlength
#         self.config.wshift = wshift
#         self.config.wshift = wshift
#         self.config.fs = samplerate

#         self._window = config.window
#         self._wsig = np.zeros(config.wlength)
#         self._wpxx = np.zeros(config.nfft)


#     def compute(self, in_data, out_data):
#         nsamples = in_data.size
#         wlength = self.config.wlength
#         novl = self.config.novl
#         nfft = self.config.nfft
#         fs = self.config.fs

#         nsegments = self.compute_nsegments(wlength, novl, nsamples)
#         wnorm = self._window.GetWindowNorm()
#         pxxnorm = 0

#         wsegm = np.zeros(wlength)
#         pxx = np.zeros(nfft)

#         segId = 0

#         while segId < nsegments:
#             sstart = segId * (wlength - novl)
#             wsegm = in_data[sstart:sstart + wlength]
#             self._window.Apply(wsegm, self._wsig)
#             self._wpxx = rfft(self._wsig)

#             pxx[0] += np.power(self._wpxx[0], 2)
#             pxx[wlength // 2] += np.power(self._wpxx[wlength // 2], 2)

#             pxx[1:wlength // 2] += (np.power(self._wpxx[1:wlength // 2], 2) +
#                                     np.power(self._wpxx[wlength // 2 + 1:wlength - 1][::-1], 2))

#             segId += 1

#         pxxnorm = (nsegments * wnorm * fs * wlength) / 2.0

#         pxx[0] /= (2.0 * pxxnorm)
#         pxx[wlength // 2] /= (2.0 * pxxnorm)
#         pxx[1:wlength // 2] /= pxxnorm

#         out_data[:] = pxx



# import numpy as np
# from typing import Optional

# def compute(self, in_signal: np.ndarray, out: np.ndarray) -> None:
#     """
#     Compute Welch's power spectral density estimate.
    
#     Args:
#         in_signal: Input signal as numpy array
#         out: Output array to store the power spectral density
#     """
    
#     nsamples = in_signal.shape[0]
#     wlength = self.config.wlength
#     novl = self.config.novl
#     nfft = self.config.nfft
#     fs = self.config.fs
    
#     nsegments = self.compute_nsegments(wlength, novl, nsamples)
#     wnorm = self._window.GetWindowNorm()
    
#     wsegm = np.zeros(wlength)
#     pxx = np.zeros(nfft)
    
#     segId = 0
    
#     while segId < nsegments:
        
#         sstart = segId * (wlength - novl)
#         wsegm = in_signal[sstart:sstart + wlength].copy()
#         self._window.Apply(wsegm, self._wsig)
        
#         # Execute real-to-real FFT (equivalent to fftw_execute_r2r)
#         self._wpxx[:] = np.fft.rfft(self._wsig, n=wlength * 2 - 2)[:wlength]
        
#         # out spans from 0 to wLength/2 (NFFT = wLength/2 + 1)
#         # ex. 	wLength  = n = 256
#         #	NFFT          = 129 (0:1:128)
#         #
#         #	out(0) 	 = wpxx(0)^2 	[Only real part - Half complex vector]
#         #	out(n/2) = wpxx(n/2)^2	[Only real part - wlenght is even - Half complex vector]
#         #	out(k) 	 = wpxx(k)^2 + wpxx(n-k)^2, k = 1 : (n/2 - 1) [Real and imagery part]
        
#         pxx[0] += np.power(self._wpxx[0], 2)
#         pxx[wlength // 2] += np.power(self._wpxx[wlength // 2], 2)
        
#         pxx[1:wlength//2] += (np.power(self._wpxx[1:wlength//2], 2) + 
#                               np.power(self._wpxx[wlength//2 + 1:wlength][::-1], 2))
        
#         segId += 1
    
#     # NORMALIZATION FACTOR
#     pxxnorm = (nsegments * wnorm * fs * wlength) / 2.0
    
#     pxx[0] = pxx[0] / (2.0 * pxxnorm)
#     pxx[wlength // 2] = pxx[wlength // 2] / (2.0 * pxxnorm)
#     pxx[1:wlength//2] = pxx[1:wlength//2] / pxxnorm
    
#     out[:] = pxx
