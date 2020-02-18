"""
This script use Gaussian-Mixture Model and Universial Background Model to do speaker verification
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fftpack import dct
import glob
import pickle

sns.set()
import os
import scipy
import librosa.display
from IPython.display import Audio
import random
from functools import reduce
from sklearn.mixture import GaussianMixture
import random

"""
1. First, select many speakers as background speaker to train Universial Background Model
For TIMIT dataset, select all train speakers in /home/hanqing/1My_Implementation_GE2E/TIMIT/TRAIN.
In total, it has 462 speakers. The TIMIT dataset is 16kHz sample rate.
"""


def extractMFCC(filename, sample_rate, nfft=512, nmels=20, nmfcc=20):
    utter_part, sr = librosa.core.load(filename, sample_rate)  # load utterance audio
    intervals = librosa.effects.split(utter_part, top_db=30)  # voice activity detection (VAD)
    S_total = []
    # 把有语音的频段挑出来，叠在一起
    for inte in intervals:
        S = librosa.core.stft(y=utter_part[inte[0]:inte[1]], n_fft=nfft)
        S = np.abs(S) ** 2
        # print("Size of S is {}".format(S.shape))
        mel_basis = librosa.filters.mel(sr=sample_rate, n_fft=nfft, n_mels=nmels)
        S = np.log10(np.dot(mel_basis, S) + 1e-6)
        # print(S.shape)
        S_total.append(S)
    # plt.show()
    # plt.pause(2)
    # Extract MFCC feature
    # mfccs = librosa.feature.mfcc(y=utter_part.astype('float'), sr=sample_rate, n_mfcc=10)
    S_total = np.concatenate(S_total, axis=1)
    # print(S_total.shape)
    mfccs = librosa.feature.mfcc(S=S_total, sr=sample_rate, n_mfcc=nmfcc)  # (128, times)
    # print(mfccs.shape)
    mfccs = mfccs.reshape(-1, nmfcc)
    return mfccs


def testModel(testPath, enrollNum):
    test_utters = glob.glob(testPath)
    speaker_en = {}
    speaker_ve = {}
    for speaker in test_utters:
        speaker_utters = glob.glob(speaker + '/*.WAV')
        # random.shuffle(speaker_utters)
        enroll = speaker_utters[:enrollNum]
        test = speaker_utters[enrollNum:]
        mfccs = []
        for en in enroll:
            mfcc = extractMFCC(en, 16000)
            mfccs.append(mfcc)
        mfccs = np.concatenate(mfccs, axis=0)
        speaker_en[speaker] = mfccs
        speaker_ve[speaker] = test
    return speaker_en, speaker_ve


if __name__ == '__main__':
    # train_utters = glob.glob('/home/hanqing/1My_Implementation_GE2E/TIMIT/TRAIN/*/*/*.WAV')
    # mfccs = []
    # total = len(train_utters)
    # print(total)
    # exit(0)
    # for idx, train_utter in enumerate(train_utters):
    #     print("{}/{}\n".format(idx, total))
    #     mfcc = extractMFCC(train_utter, 16000)
    #     # mfcc = mfcc - np.mean(mfcc, axis=0)
    #     mfccs.append(mfcc)
    # mfccs = np.concatenate(mfccs, axis=0)
    # np.save('ubmdata', mfccs)
    # print("Train Speaker Data Saved! ")
    # exit(0)

    # mfccs = np.load('ubmdata.npy')
    # print("Train Speaker Data Loaded! ")
    # UBM = GaussianMixture(n_components=56, covariance_type='full', max_iter=100)
    # UBM.fit(mfccs)
    # print(UBM)
    # pickle.dump(UBM, open("ubm.p", "wb"))

    UBM = pickle.load(open("ubm.p", "rb"))

    # print(UBM)
    speaker_en, speaker_ve = testModel('/home/hanqing/1My_Implementation_GE2E/TIMIT/TEST/*/*', 3)

    # test_utters = glob.glob('/home/hanqing/1My_Implementation_GE2E/TIMIT/TEST/*/*')
    fit_speaker = speaker_en['/home/hanqing/1My_Implementation_GE2E/TIMIT/TEST/DR6/MRJS0']
    ori_pred = UBM.predict(fit_speaker)
    print(ori_pred)
    print(ori_pred.shape)
    UBM.fit(fit_speaker)
    pred = UBM.predict(fit_speaker)
    print(pred)
    print(pred.shape)