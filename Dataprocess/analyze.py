import numpy as np
import sklearn.metrics as metrics
from sklearn.metrics import classification_report

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns


def plot_pie(tr_df, ts_df, cls2lbl):

    tr_distributions = tr_df['label'].value_counts().sort_index().tolist()
    ts_distributions = ts_df['label'].value_counts().sort_index().tolist()
    
    plt.figure(figsize=(13, 13))
    plt.subplot(1, 2, 1)

    color = sns.color_palette('pastel')[:len(list(cls2lbl.keys()))]
    patches, _ = plt.pie(tr_distributions, colors=color, startangle=90, radius=1.2)
    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(np.array(list(cls2lbl.keys())), (np.array(tr_distributions) / len(tr_df)) * 100)]
    plt.title('Class Distribution in Train Set')

    patches, labels, _ =  zip(*sorted(zip(patches, labels, tr_distributions), key=lambda x: x[2], reverse=True))
    plt.legend(patches, labels, loc='upper left', bbox_to_anchor=(-0.1, 1.), fontsize=8)

    plt.subplot(1, 2, 2)

    color = sns.color_palette('pastel')[:len(list(cls2lbl.keys()))]
    patches, _ = plt.pie(ts_distributions, colors=color, startangle=90, radius=1.2)
    labels = ['{0} - {1:1.2f} %'.format(i,j) for i,j in zip(np.array(list(cls2lbl.keys())), (np.array(ts_distributions) / len(ts_df)) * 100)]
    plt.title('Class Distribution  in Test Set')

    patches, labels, _ =  zip(*sorted(zip(patches, labels, ts_distributions), key=lambda x: x[2], reverse=True))

    plt.legend(patches, labels, loc='upper left', bbox_to_anchor=(-0.1, 1.), fontsize=8)
    plt.show()


def fetch_examples(tr_df, lbl2cls, data_transforms, device):
    sample_records = []
    num_classes = 4
    for label in range(num_classes):
        class_addr = tr_df[tr_df['label'] == label].iloc[10]
        img = Image.open(class_addr.img_id).convert('RGB')
        lbl = class_addr.label
        sample_records.append((img, lbl))

    sample_recs = []
    random_indices = np.random.choice(range(num_classes), num_classes, replace=False)

    fig, axes = plt.subplots(1, num_classes, figsize=(15, 5))


    for i, idx in enumerate(random_indices):
        image, label = sample_records[idx]
        image_tr = data_transforms['train'](image)
        sample_recs.append((image, image_tr.to(device), label))
        image_tr = image_tr.permute(1, 2, 0)
        axes[i].imshow(np.uint8(np.array(image_tr)))
        axes[i].set_title(f"Tumor: {lbl2cls[label]}")
        axes[i].axis('off')

    plt.show()


def evaluation_report(y, y_pred):
    print(classification_report(y, y_pred))


def plot_heatmap(y, y_pred, classes):
    conf_mat = metrics.confusion_matrix(y, y_pred)
    _, ax = plt.subplots(figsize=(7,7))
    s = sns.heatmap(conf_mat,
                square=True,
                annot=True,
                fmt='d',
                cbar=False,
                xticklabels=classes,
                yticklabels=classes, ax=ax)
    s.set_xlabel('Ground Truth')
    s.set_ylabel('Predicted')
    plt.show()


def plot_loss_acc(history, figsize=(12,6)):
    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.plot(history[0]['train'], label='Train Loss')
    plt.plot(history[0]['test'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history[1]['train'], label='Train Accuracy')
    plt.plot(history[1]['test'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy in %')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    import preprocessing
    dfs, cls2lbl, _ = preprocessing.preprocess()
    plot_pie(dfs['train'], dfs['test'], cls2lbl)