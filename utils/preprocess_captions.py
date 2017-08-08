import os
import re
import tqdm
import json
import time
from numpy import *

def main(captions_train, captions_val, output_path, threshold=5, target_length=20):
    ''''''
    j_train = json.loads(open('/home/paperspace/data/ms_coco/captions_train2014.json').read())
    j_val = json.loads(open('/home/paperspace/data/ms_coco/captions_val2014.json').read())
    train = {d['caption']: d['image_id'] for d in j_train['annotations']}
    val = {d['caption']: d['image_id'] for d in j_val['annotations']}

    print('filtering non-alphabetic characters'); time.sleep(.2)
    for d in (train, val):
        for k in tqdm.tqdm(d.keys()):
            f = re.sub('[^a-z \h]', ' ', k.lower())
            d[f] = d.pop(k)

    print('splitting on spaces and hashing'); time.sleep(.2)
    total = train.keys() + val.keys()
    instances = {}
    for caption in tqdm.tqdm(total):
        for w in caption.split(' '):
            if w != '':
                try:
                    instances[w] += 1
                except:
                    instances[w] = 1

    print('filtering stopwords')
    words = sorted(instances.keys())
    words_set = set(words)
    go_words = set([w for (w, c) in instances.items() if c > threshold])
    stop_words = words_set - go_words

    words_to_inds = dict(zip(sorted(list(go_words)), range(len(go_words))))
    inds_to_words = dict(zip(range(len(go_words)), sorted(list(go_words))))

    words_to_inds['UNKNOWN'] = len(go_words)
    inds_to_words[len(go_words)] = 'UNKNOWN'


    print('converting captions to indices'); time.sleep(.2)
    for d in (train, val):
        for k in tqdm.tqdm(d.keys()):
            v = d.pop(k)
            split = k.split(' ')
            split_caption = []
            for w in split:
                if w != '':
                    if w in stop_words:
                        w = 'UNKNOWN'
                    split_caption.append(w)

            indices = [words_to_inds[w] for w in split_caption]
            try:
                d[v].append(indices)
            except:
                d[v] = []
                d[v].append(indices)
                
    n = len(words_to_inds) + 1
    words_to_inds['SOS'] = n-1
    words_to_inds['EOS'] = n
    inds_to_words[n-1] = 'SOS'
    inds_to_words[n] = 'EOS'

    for d in (train, val):
        for k, v in d.items():
            n_captions = len(v)
            if n_captions < 7:
                d[k] += [[]]*(7-n_captions)
            for i, c in enumerate(d[k]):
                v[i] = c[:target_length-2]
                v[i] = [n-1] + v[i] + [n]
                if len(v[i]) < target_length:
                    v[i] += [-1]*(target_length-len(v[i]))
                    
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(output_path, 'train_captions.json'), 'w') as outfile:
        json.dump(train, outfile)
    with open(os.path.join(output_path, 'val_captions.json'), 'w') as outfile:
        json.dump(val, outfile)
    with open(os.path.join(output_path, 'words_to_inds.json'), 'w') as outfile:
        json.dump(words_to_inds, outfile)
    with open(os.path.join(output_path, 'inds_to_words.json'), 'w') as outfile:
        json.dump(inds_to_words, outfile)
        
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('captions_train', type=str, 
        help='json containing training captions')
    p.add_argument('captions_val', type=str, 
        help='json containing validation captions')
    p.add_argument('output_path', type=str, 
        help='path in which to save processed captions')
    d = p.parse_args().__dict__
    main(**d)