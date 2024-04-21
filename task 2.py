import h5py
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
plt.rcParams['font.size'] = 12
import matplotlib.mlab as mlab
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

def whiten(Strain, psdInterp, Xspacing):
  StrainLen = len(Strain)
  freqs = np.fft.rfftfreq(StrainLen, Xspacing)
  StrainFreq = np.fft.rfft(Strain)
  NORM = 1. / np.sqrt(1./(Xspacing*2))
  StrainFreqWhiten = StrainFreq / np.sqrt(psdInterp(freqs)) * NORM
  StrainWhiten = np.fft.irfft(StrainFreqWhiten, n=StrainLen)
  return StrainWhiten

dataFile = "H-H1_GWOSC_4KHZ_R1-1242459842-32.hdf5"
templateFile = "GW150914_4_template.hdf5"
with h5py.File(dataFile, 'r') as data:
  Strain   = data['strain']['Strain'][...]
  GPSstart = data['meta']['GPSstart'][()]
  Xspacing = data['strain']['Strain'].attrs['Xspacing']
  Duration = data['meta']['Duration'][()]
with h5py.File(templateFile, 'r') as data:
  template = data['template'][...]
  template = template[0] + template[1]*1.j

timeSeries = np.arange(GPSstart, GPSstart+Duration, Xspacing)
Fs   = int(1./Xspacing)
NFFT = 4*Fs
NOVL = int(NFFT/2)
blkmWin  = np.blackman(NFFT)
tukeyWin = signal.windows.tukey(template.size, alpha=1./8)
templateFreq = np.fft.fftfreq(template.size) * Fs
df = templateFreq[1] - templateFreq[0]

templateFFT = np.fft.fft(template*tukeyWin) / Fs
StrainFFT = np.fft.fft(Strain*tukeyWin) / Fs
StrainPSD, freqs = mlab.psd(
  Strain, Fs=Fs, NFFT=NFFT, window=blkmWin, noverlap=NOVL
)
powerVec = np.interp(np.abs(templateFreq), freqs, StrainPSD)
optimalSNR = StrainFFT * templateFFT.conjugate() / powerVec
optimalSNRTime = 2 * np.fft.ifft(optimalSNR) * Fs
normFac = 1 * (templateFFT*templateFFT.conjugate()/powerVec).sum() * df
normFac = np.sqrt(np.abs(normFac))
SNRComplex = optimalSNRTime / normFac
peakIdx = int(Strain.size/2)
SNRComplex = np.roll(SNRComplex, peakIdx)
SNR = abs(SNRComplex)

idxMax = np.argmax(SNR)
timeMax = timeSeries[idxMax]

plt.figure(figsize=(10,3))
plt.subplot(1,2,1)
plt.ylabel("SNR")
plt.plot(timeSeries-timeMax, SNR, 'r', label="H1 SNR")
plt.xlabel("time (s) since {0:.4f}".format(timeMax))
plt.legend(loc='upper left')

plt.subplot(1,2,2)
plt.plot(timeSeries-timeMax, SNR, 'r', label="H1 SNR")
plt.xlim([-0.1,0.1])
plt.xticks([-0.1, -0.05, 0, 0.05, 0.1])
plt.xlabel("time (s) since {0:.4f}".format(timeMax))
plt.ylabel("SNR")
plt.legend(loc='upper left')
plt.grid('on')
# plt.savefig("GWDataAnalysis-SNR.pdf", bbox_inches='tight')
plt.show()


freqBand = [40.0, 210.0]
b, a = butter(4, [freqBand[0]*2./Fs,freqBand[1]*2./Fs], btype='band')
NORM = np.sqrt((freqBand[1]-freqBand[0]) / (Fs/2))

StrainPSD, freqs = mlab.psd(Strain, Fs=Fs, NFFT=NFFT)
StrainPSDInterp = interp1d(freqs, StrainPSD)
StrainWhiten = whiten(Strain, StrainPSDInterp, Xspacing)
StrainWhitenBP = filtfilt(b, a, StrainWhiten) / NORM
SNRmax = SNR[idxMax]
phase = np.angle(SNRComplex[idxMax])
offset = (idxMax-peakIdx)
templatePhaseshifted = np.real(template*np.exp(1.j*phase))
templateRolled = np.roll(templatePhaseshifted,offset) / (normFac/SNRmax)
templateWhitened = whiten(templateRolled, interp1d(freqs, StrainPSD), Xspacing)
templateMatch = filtfilt(b, a, templateWhitened) / NORM

plt.figure(figsize=(10,3))
plt.plot(timeSeries-timeMax, StrainWhitenBP, 'r', label="H1 whitened $h(t)$")
# plt.plot(timeSeries-timeMax, templateMatch, 'k', label="Template")
plt.xlim([-0.15,0.05])
plt.ylim([-10,10])
plt.grid('on')
plt.xlabel("Time since {0:.4f}".format(timeMax))
plt.ylabel("whitened strain")
plt.legend(loc='upper left')
# plt.savefig("GWDataAnalysis-Match.pdf", bbox_inches='tight')
plt.show()