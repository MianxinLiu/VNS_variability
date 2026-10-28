"""Compute motor SE/DFA at the native 1-kHz neural sampling rate.

The frequency and DFA scale specifications match the EEG analysis: the neural
signal is band-pass filtered at 0.5--80 Hz, the spectral exponent is fitted at
1--40 Hz, and DFA uses 10--2000-ms windows in 100-ms increments.
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.signal import butter, sosfiltfilt, detrend, welch
import fathon
from fathon import fathonUtils as fu
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from compute_spectral_exponent import compute_SpectralExponent

def main():
    archive=np.load(sys.argv[1]); output=[]
    fs=1000
    sos=butter(4,[.5,80],btype='bandpass',fs=fs,output='sos')
    windows=fu.linRangeByStep(10,2000,100)
    for raw in archive['S_E']:
        filtered=sosfiltfilt(sos,raw.astype(float),axis=1)
        slopes=[];dfas=[]
        for signal,filtered_signal in zip(raw,filtered):
            dfa=fathon.DFA(fu.toAggregated(detrend(filtered_signal)))
            dfa.computeFlucVec(windows,revSeg=True,polOrd=3)
            dfas.append(float(dfa.fitFlucVec()[0]))
            freq,psd=welch(filtered_signal,fs=fs,nperseg=fs*3,
                           noverlap=fs*2,detrend='linear')
            band=(freq>=1)&(freq<=40)
            slope,*_=compute_SpectralExponent(freq[band],psd[band],do_plot=False)
            slopes.append(float(slope))
        output.append(dict(SE=float(np.mean(slopes)),DFA=float(np.mean(dfas))))
    Path(sys.argv[2]).write_text(json.dumps(output,indent=2))

if __name__=='__main__':main()
