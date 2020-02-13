import os
import os.path as osp
import csv

cur_dir = osp.dirname(osp.abspath(__file__))
root = osp.dirname(cur_dir)
wav_path = osp.join(root, "wav")
speakers = os.listdir(wav_path)
speakers = sorted(speakers)
speakers_utters = {}
for i in range(100):
    speaker = osp.join(wav_path, speakers[i])
    utters_folder = os.listdir(speaker)
    speaker_utters = []
    for utters in range(len(utters_folder)):
        fold = osp.join(speaker, utters_folder[utters])
        for utter in os.listdir(fold):
            speaker_utters.append(osp.join(fold,utter))
    speakers_utters[i] = speaker_utters

with open('test_list_new.csv', 'w') as cs:
    fieldnames = ['filename', 'speaker']
    writer = csv.DictWriter(cs, fieldnames=fieldnames)
    writer.writeheader()
    keys = speakers_utters.keys()
    total_rows = 0
    for key in keys:
        # total_rows += len(speakers_utters[key])
        for file in speakers_utters[key]:
            writer.writerow({'filename':file, 'speaker':key+1})
