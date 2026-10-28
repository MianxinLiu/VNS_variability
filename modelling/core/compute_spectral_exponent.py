import numpy as np
from scipy.stats import linregress
from scipy.signal import find_peaks

def compute_SpectralExponent(XX, YY, do_plot=False):
    '''
    Fits a line on log-frequency vs log-power spectral density (PSD), excluding peaks and their base.

    Parameters
    ----------
    XX (array-like): Frequency bins of interest (Hz).
    YY (array-like): PSD relative to the bins of interest (muV^2 /Hz).
    do_plot (bool, optional): Retained for compatibility; plotting is handled in plotting/.

    Returns
    -------
    slope (float): Slope of the second (final) power-law fit, i.e., the spectral exponent.
    intercept (float): Intercept of the second (final) power-law fit.
    stats (dict): Dictionary containing fitting statistics including slopes (slope), intercepts (inter), correlation coefficients (r),
                  and root mean squared error of residuals (sigma) for both the initial and final fits. 
                  Stats from the initial fit are indicated with a 0 at the end (e.g., slope0, sigma0, etc.).
                  threResPks refers to the threshold used to find large peaks (excluded from the final fit).
    vectors (dict): Dictionary containing vectors of resampled frequencies (f_resamp), observed and predicted PSD, residuals (resid for the final fit and resid0 for the initial fit),
                    and boolean array indicating rejected data points due to peaks (reject).

    Usage Example
    -------------
    from scipy.signal import welch
    def dsearchn(x, value):
        dist_from_value = np.abs(x-value)
        return np.where(dist_from_value == dist_from_value.min())[0][0]
    # eeg_1chan is one eeg channel after preprocessing and filtering (be aware of how filters impact the psd)
    # if high pass filter (butter order 5) is set at 0.5 Hz, then begin fitting at 1 Hz
    # if low pass filter (butter order 5) is set at 60 Hz, then stop fitting at approximately 45 Hz. Better even no low pass
    fs=1000 # use the sampling rate from your data (in Hz)
    frex, psd_1chan = welch(eeg_1chan, fs=fs, nperseg=fs*3, noverlap=fs*2, detrend='linear')
    frband = [1, 40]
    frBins = np.arange(dsearchn(frex, frband[0]), dsearchn(frex, frband[1])+1)
    XX = frex[frBins]
    YY = psd_1chan[frBins]
    slope, inter, stats, vectors = compute_SpectralExponent(XX, YY)
    '''
    X = np.log10(XX)
    Y = np.log10(YY)
    
    # UPSAMPLE AND INTERPOLATE IN LOG-LOG:  X, Y --> Xi, Yi
    XXi = np.logspace(X[0], X[-1],len(X)*4)
    Xi = np.log10(XXi)
    Yi = np.interp(Xi, X, Y,)
    YYi = 10**(Yi)
    
    # STEP 1, FIT 1st LINE
    linfit0 = linregress(Xi, Yi) # alternative = 'less'???
    slope0, intercept0 = linfit0[0], linfit0[1]
    YPred0 = Xi*slope0+intercept0
    YRes0 = Yi-YPred0
    sigma0 = np.sqrt(np.mean(YRes0**2))
    
    # FIND DEVIANT RESIDUALS
    threRes = 0
    boolYdev = YRes0 > threRes
    idxs_dev = np.argwhere(np.diff(boolYdev,  prepend=False, append=False)).reshape(-1,2)
    
    # Find peaks to be excluded next
    idxs_peaks = find_peaks(Yi)[0]
    threResPks= np.median(np.absolute(YRes0 - np.median(YRes0))) * 1
    
#     threResPks= max(threResPks, .1) # this is a heuristc value based on typical psd from eeg with unit of microV^2/Hz
    
    idxs_peaks = idxs_peaks[YRes0[idxs_peaks] > threResPks ]
    boolYdev2 = np.zeros_like(boolYdev)
    for idx_start, idx_end in idxs_dev:
        range_dev = np.arange(idx_start, idx_end)
        # where there are peaks with large residuals, set 1 in the boolean used for subsequent exclusion of peaks
        boolYdev2[range_dev] = np.any(np.intersect1d(idxs_peaks, range_dev)) 
        
    # STEP 2 , FIT 2nd line after excluding peaks
    linfit = linregress(Xi[~boolYdev2], Yi[~boolYdev2])
    slope, intercept = linfit[0], linfit[1]
    YPred = Xi*slope+intercept
    YRes = Yi[~boolYdev2]-YPred[~boolYdev2]
    sigma = np.sqrt(np.mean(YRes**2))
    
    # store the results
    stats = dict(slope=slope, slope0=slope0,  
                 inter=intercept, inter0=intercept0,
                 r=linfit[2], r0=linfit0[2], 
                 sigma=sigma, sigma0=sigma0, 
                 threResPks=threResPks)
    vectors = dict(f_resamp=XXi, observed=YYi, predicted=YPred,
                   resid=YRes, resid0=YRes0, reject=boolYdev2)
    
    return slope, intercept, stats, vectors
