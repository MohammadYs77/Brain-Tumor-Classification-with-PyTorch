import os
import pandas as pd

addrs = ['D:/AI Courses/Semester 4/Brain Tumor/brain_tumor_dataset/Training',
                'D:/AI Courses/Semester 4/Brain Tumor/brain_tumor_dataset/Testing']


def preprocess():
    
    classes = os.listdir(addrs[0])
    labels = list(range(len(classes)))

    cls2lbl = dict(zip(classes, labels))
    lbl2cls = {v: k for k, v in cls2lbl.items()}
    
    dfs = {'train': None, 'test': None}
    
    for addr in addrs:
        
        all_imgs_dirs = []
        all_lbls = []

        for dirname, _, filenames in os.walk(addr):
            for filename in filenames:
                all_imgs_dirs.append(os.path.join(dirname, filename))

        for addr in all_imgs_dirs:
            for clss in cls2lbl.keys():
                if clss in addr:
                    all_lbls.append(cls2lbl[clss])

        key = 'train' if 'train' in addr.lower() else 'test'
        dfs[key] = pd.DataFrame({'img_id': all_imgs_dirs, 'label': all_lbls})
        dfs[key] = dfs[key].sample(frac=1).reset_index(drop=True)
    
    return dfs, cls2lbl, lbl2cls