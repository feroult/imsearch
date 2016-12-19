from __future__ import print_function
import sys
import cv2
import numpy as np
import json
import os.path

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def load_dataset(file_name):
    dataset = {}
    with open('/imsearch/data/private/look-nice/' + file_name) as f:
        for line in f:
            j = json.loads(line)
            key = j['key']['path'][0]['name']
            dataset[key] = j
    return dataset


# datasetsw
visions = load_dataset('visions_dataset.bak')
images = load_dataset('images.bak')

# create bow
surf = cv2.xfeatures2d.SURF_create()
extract = cv2.xfeatures2d.SURF_create()

matcher = cv2.BFMatcher(cv2.NORM_L2)
BOW = cv2.BOWKMeansTrainer(100)

for key in visions:
    filename = '/imsearch/data/private/look-nice/imagens/' + images[key]['properties']['filename']['stringValue']
    image = cv2.imread(filename, 0)
    eprint("file", filename)
    try:
        kp, des = surf.detectAndCompute(image, None)
        BOW.add(des)
    except:
        pass

dictionary = BOW.cluster()

bow_extract = cv2.BOWImgDescriptorExtractor(extract, matcher)
bow_extract.setVocabulary( dictionary )

# generate feature vectors
for key in visions:
    filename = '/imsearch/data/private/look-nice/imagens/' + images[key]['properties']['filename']['stringValue']
    image = cv2.imread(filename, 0)
    kp = surf.detect(image)
    inp = bow_extract.compute(image, kp)

    try:
        statusStr = visions[key]['properties']['status']['stringValue']
        status = '0,' if statusStr == 'TRASH' else '1,'
        print(status + ','.join(map(str, inp[0])))
    except:
        pass
