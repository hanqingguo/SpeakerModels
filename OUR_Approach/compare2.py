from utils import *
import webrtcvad


def extract_feature(filename, lower_bound, upper_bound, melfilterSampleRate, nfft, nmels, SampleRate, nmfcc, log=True):
    sr, samples = readfile(filename)
    Xdb = get_stft(sr, samples)
    max_idx = get_fri_indics(Xdb, lower_bound, upper_bound)
    mel_basis = gen_mel(melfilterSampleRate, nfft, nmels)
    mfcc = get_mfcc(mel_basis, max_idx, Xdb, SampleRate, nmfcc, log)
    return mfcc, max_idx


if __name__ == '__main__':
    filename1 = '../ultraData/10/good/8.wav'
    lower_bound = 200
    upper_bound = 800
    melfilterSampleRate = 48000  # 48kHz to expose more high frequency feature
    nfft = 2048  # number of frequency bins is 1024
    nmels = 10
    SampleRate = 192000  # high frequency ranage
    nmfcc = 10
    mfcc1, max_idx1 = extract_feature(filename1, lower_bound, upper_bound, melfilterSampleRate, nfft, nmels, SampleRate,
                                      nmfcc,
                                      log=True)
    # TODO: APPLY VAD TO DETECT ACTIVE VOICE PART
    # vad = webrtcvad.Vad
    # vad.set_mode(1)
    # vad.is_speech()
    filename2 = '../ultraData/3/good/7.wav'
    mfcc2, max_idx2 = extract_feature(filename2, lower_bound, upper_bound, melfilterSampleRate, nfft, nmels, SampleRate,
                                      nmfcc,
                                      log=True)
    fricative1 = mfcc1[1:, max_idx1].reshape(-1, mfcc1.shape[0] - 1)
    fricative2 = mfcc2[1:, max_idx2].reshape(-1, mfcc2.shape[0] - 1)
    cos = cal_cos(fricative1, fricative2)
    print(cos)
