import sox
import os
import os.path as osp
import glob

tfm = sox.Transformer()
tfm.convert(samplerate=16000)
curdir = os.getcwd()
speakers = glob.glob(curdir+"/[!d]*")
for speakers 


print(speakers)


# tfm.build('3/bad/1.wav', 'test.wav')