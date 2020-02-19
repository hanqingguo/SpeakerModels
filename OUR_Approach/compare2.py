from utils import *
import matplotlib.pyplot as plt
import numpy as np
import glob


def extract_feature(filename, settings):
    sr, samples = readfile(filename)
    Xdb = get_stft(sr, samples)
    max_idx = get_fri_indics(Xdb, settings['lower_bound'], settings['upper_bound'])
    mel_basis = gen_mel(settings['melfilterSampleRate'], settings['nfft'], settings['nmels'])
    mfcc = get_mfcc(mel_basis, max_idx, Xdb, settings['SampleRate'], settings['nmfcc'], settings['log'])
    return mfcc, max_idx


def cosTwo(filename1, filename2, settings):
    mfcc1, max_idx1 = extract_feature(filename1, settings)
    mfcc2, max_idx2 = extract_feature(filename2, settings)
    fricative1 = mfcc1[1:, max_idx1].reshape(-1, mfcc1.shape[0] - 1)
    fricative2 = mfcc2[1:, max_idx2].reshape(-1, mfcc2.shape[0] - 1)
    cos = cal_cos(fricative1, fricative2)
    return cos


if __name__ == '__main__':
    filename1 = '../ultraData/8/bad/10.wav'
    # settings = {
    #     'lower_bound': 200,
    #     'upper_bound': 800,
    #     'melfilterSampleRate': 48000,  # 48kHz to expose more high frequency feature
    #     'nfft': 2048,  # number of frequency bins is 1024
    #     'nmels': 10,
    #     'SampleRate': 192000,  # high frequency ranage
    #     'nmfcc': 10,
    #     'log': True
    # }
    # # mfcc1, max_idx1 = extract_feature(filename1, settings)
    # # # TODO: APPLY VAD TO DETECT ACTIVE VOICE PART
    # # # vad = webrtcvad.Vad
    # # # vad.set_mode(1)
    # # # vad.is_speech()
    # # filename2 = '../ultraData/5/bad/1.wav'
    # # mfcc2, max_idx2 = extract_feature(filename2, settings)
    # #
    # # fricative1 = mfcc1[1:, max_idx1].reshape(-1, mfcc1.shape[0] - 1)
    # # fricative2 = mfcc2[1:, max_idx2].reshape(-1, mfcc2.shape[0] - 1)
    # # cos = cal_cos(fricative1, fricative2)
    # # print(cos)
    # # cos = cosTwo('../ultraData/5/bad/1.wav', '../ultraData/6/bad/1.wav', settings)
    # # print(cos)
    # # exit(0)
    #
    # cosMatrix = np.zeros((26, 26))
    # speakerID = '3'
    # microphone = 'bad'
    # files = glob.glob('../ultraData/' + speakerID + '/' + microphone + '/*.wav')
    #
    # con_speakerID = '6'
    # con_microphone = 'bad'
    # con_files = glob.glob('../ultraData/' + con_speakerID + '/' + microphone + '/*.wav')
    # # sort them by 1.wav, 2.wav ... rather than 1.wav, 10.wav, ...
    # # print(sorted(files, key=lambda x: int(x.split('/')[-1][:-4])))
    # files = sorted(files, key=lambda x: int(x.split('/')[-1][:-4]))
    # con_files = sorted(con_files, key=lambda x: int(x.split('/')[-1][:-4]))
    # for refidx, reffile in enumerate(files):
    #     for idx, file in enumerate(con_files):
    #         print("refidx is : {}\n idx is : {}".format(refidx, idx))
    #         cos = cosTwo(reffile, file, settings)
    #         cosMatrix[refidx][idx] = cos
    # np.save('speaker3vs6-original', cosMatrix)

    cosMatrix = np.load('speaker6-original.npy')
    print(cosMatrix)
    fig, ax = plt.subplots()
    ax.matshow(cosMatrix)

    for (i, j), z in np.ndenumerate(cosMatrix):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')
    plt.show()