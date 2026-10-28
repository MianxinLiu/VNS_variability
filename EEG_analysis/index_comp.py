import mne
import os
import matplotlib
from compute_spectral_exponent import compute_SpectralExponent
import EntropyHub as EH
import fathon
from fathon import fathonUtils as fu
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from scipy.signal import welch, detrend


def dsearchn(x, value):
    dist_from_value = np.abs(x - value)
    return np.where(dist_from_value == dist_from_value.min())[0][0]

workpath = '/mnt/pci-0000:00:17.0-ata-6/VNS/project/processed_eeg_ave_ar2/'
sub = os.listdir(workpath)

# spectral exponents
for sid in range(len(sub)):
    file = os.listdir(os.path.join(workpath, sub[sid]))
    outpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cohort2"))

    days = np.zeros(len(file))
    SE = np.zeros([len(file), 256])

    for fid in range(len(file)):
        epochs = mne.read_epochs(os.path.join(workpath, sub[sid], file[fid]), preload=True)
        days[fid] = int(file[fid].split('w_')[0])

        try:
            epochs_ec = epochs['eyec']
            # epochs_eo = epochs['eyeo']
        except:
            epochs_ec = epochs['LOFF']
            # epochs_eo = epochs['L-ON']

        for elc in range(256):
            eeg_1chan = epochs_ec.get_data()
            fs = 1000  # use the sampling rate from your data (in Hz)
            frex, psd_1chan = welch(eeg_1chan[:, elc, :], fs=fs, nperseg=fs * 3, noverlap=fs * 2, detrend='linear')
            psd_1chan = psd_1chan.mean(axis=0)
            frband = [1, 40]
            frBins = np.arange(dsearchn(frex, frband[0]), dsearchn(frex, frband[1]) + 1)
            XX = frex[frBins]
            YY = psd_1chan[frBins]
            slope, inter, stats, vectors = compute_SpectralExponent(XX, YY, do_plot=False)
            SE[fid, elc] = slope

    os.makedirs(os.path.join(outpath, sub[sid]), exist_ok=True)
    index = np.argsort(days)
    np.save(os.path.join(outpath, sub[sid], 'days.npy'), days[index])
    np.save(os.path.join(outpath, sub[sid], 'SE.npy'), SE[index,:])

    import matplotlib.pyplot as plt
    mean_SE = np.mean(SE, axis=1)
    plt.figure(figsize=(10, 6))
    index = np.argsort(days)
    days, mean_SE = days[index], mean_SE[index]
    plt.plot(days, mean_SE)
    plt.xlabel('Weeks')
    plt.ylabel('Averaged SE')
    plt.title('Changes')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(os.path.join(outpath, sub[sid], 'change_SE.png'))
    plt.close()

# DFA
for sid in range(len(sub)):
    file = os.listdir(os.path.join(workpath, sub[sid]))
    outpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cohort2"))

    days = np.zeros(len(file))
    Hurst = np.zeros([len(file), 256])
    eponum = np.zeros(len(file))

    for fid in range(len(file)):
        epochs = mne.read_epochs(os.path.join(workpath, sub[sid], file[fid]), preload=True)
        days[fid] = int(file[fid].split('w_')[0])

        try:
            epochs_ec = epochs['eyec'].get_data()
            # epochs_eo = epochs['eyeo'].get_data()
        except:
            epochs_ec = epochs['LOFF'].get_data()
            # epochs_eo = epochs['L-ON'].get_data()

        eponum[fid] = epochs_ec.shape[0]
        Htemp = np.zeros([epochs_ec.shape[0], 256])
        for epo in range(epochs_ec.shape[0]):
            H = np.zeros(256)
            for ele in range(256):
                signal = detrend(np.squeeze(epochs_ec[epo, ele, :].T))
                a = fu.toAggregated(signal)
                # initialize dfa object
                pydfa = fathon.DFA(a)
                # compute fluctuation function and Hurst exponent
                wins = fu.linRangeByStep(10, 2000, 100)
                n, F = pydfa.computeFlucVec(wins, revSeg=True, polOrd=3)
                DFA, H_intercept = pydfa.fitFlucVec()
                H[ele] = DFA

            Htemp[epo, :] = H
        Hurst[fid, :] = np.mean(Htemp, axis=0)

    os.makedirs(os.path.join(outpath, sub[sid]), exist_ok=True)
    index = np.argsort(days)
    np.save(os.path.join(outpath, sub[sid], 'Weeks.npy'), days[index])
    np.save(os.path.join(outpath, sub[sid], 'DFA.npy'), Hurst[index, :])

# multiscale entropy
Mobj = EH.MSobject('DispEn')

for sid in range(len(sub)):
    file = os.listdir(os.path.join(workpath, sub[sid]))
    outpath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cohort2"))

    days = np.zeros(len(file))
    MSE = np.zeros([len(file), 256, 20])

    for fid in range(len(file)):
        epochs = mne.read_epochs(os.path.join(workpath, sub[sid], file[fid]), preload=True)
        days[fid] = int(file[fid].split('w_')[0])

        try:
            epochs_ec = epochs['eyec']
            # epochs_eo = epochs['eyeo']
        except:
            epochs_ec = epochs['LOFF']
            # epochs_eo = epochs['L-ON']

        eeg_1chan = epochs_ec.get_data()
        for elc in range(256):
            MSEtemp = np.zeros([eeg_1chan.shape[0], 20])
            for epo in range(eeg_1chan.shape[0]):
                MSEtemp[epo, :], CI = EH.MSEn(np.squeeze(eeg_1chan[epo, elc, :]), Mobj, Scales=20)
            MSE[fid, elc, :] = MSEtemp.mean(axis=0)

    os.makedirs(os.path.join(outpath, sub[sid]), exist_ok=True)
    index = np.argsort(days)
    # np.save(os.path.join(outpath, sub[sid], 'days.npy'), days[index])
    np.save(os.path.join(outpath, sub[sid], 'MSE.npy'), MSE[index,:])
