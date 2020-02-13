import glob
import os
import csv
import random

enrollNum = 3
speakers = glob.glob('/home/hanqing/1ASpeakerRecognition/ultraData/[!d]*')
# folder = "test_model"
folder = "ultraSound"
good_enroll = open('./vggvox-speaker-identification/'+folder+'/good_enroll'+str(enrollNum)+'.csv', 'w')
good_test = open('./vggvox-speaker-identification/'+folder+'/good_test'+str(enrollNum)+'.csv', 'w')
bad_enroll = open('./vggvox-speaker-identification/'+folder+'/bad_enroll'+str(enrollNum)+'.csv', 'w')
bad_test = open('./vggvox-speaker-identification/'+folder+'/bad_test'+str(enrollNum)+'.csv', 'w')

fieldnames = ['filename', 'speaker']


wgood_enroll = csv.DictWriter(good_enroll, fieldnames=fieldnames)
wgood_enroll.writeheader()
wgood_test = csv.DictWriter(good_test, fieldnames=fieldnames)
wgood_test.writeheader()
wbad_enroll = csv.DictWriter(bad_enroll, fieldnames=fieldnames)
wbad_enroll.writeheader()
wbad_test = csv.DictWriter(bad_test, fieldnames=fieldnames)
wbad_test.writeheader()


print(speakers)
for speaker in speakers:
    good_utters = glob.glob(speaker +'/good/*.wav')
    bad_utters = glob.glob(speaker + '/bad/*.wav')
    speakersid = speaker.split('/')[-1]
    random.shuffle(good_utters)
    random.shuffle(bad_utters)
    en_good = good_utters[:enrollNum]
    test_good = good_utters[enrollNum:]
    en_bad = bad_utters[:enrollNum]
    test_bad = bad_utters[enrollNum:]
    for i in range(enrollNum):
        wgood_enroll.writerow({'filename':en_good[i], 'speaker':speakersid})
        # wgood_enroll.writerow({'filename': 'hello', 'speaker': 'hi'})
        wbad_enroll.writerow({'filename':en_bad[i], 'speaker':speakersid})
    for i in range(len(test_good)):
        wgood_test.writerow({'filename':test_good[i], 'speaker':speakersid})
    for i in range(len(test_bad)):
        wbad_test.writerow({'filename':test_bad[i], 'speaker':speakersid})
