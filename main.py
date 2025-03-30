from Dataprocess import preprocessing, analyze
from Pipelines import img_aug_pipeline, data_pipeline
import model

import torch
import torch.nn as nn
import torch.optim as optim
import argparse


def main(args):

    device = torch.device('cpu' if args['device'] else 'cuda')
    BATCH_SIZE = args['batch']
    EPOCH = args['epochs']
    RESIZE = args['img_resize']
    addrs = ['/brain_tumor_dataset/Training', '/brain_tumor_dataset/Testing']
    
    
    dfs, cls2lbl, lbl2cls = preprocessing.preprocess(addrs)
    data_transform = img_aug_pipeline.make_augmentation_pipeline(RESIZE)
    
    analyze.fetch_examples(dfs['train'], lbl2cls, data_transform, device=device)
    dataloaders = data_pipeline.build_dataloader(
        dfs['train'],
        dfs['test'],
        resize=RESIZE,
        batch=BATCH_SIZE,
        data_transforms=data_transform)

    mdl = model.Model(3).to(device)
    opt = optim.Adam(mdl.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()
    print()

    model_trained, history = mdl.fit(dataloaders, criterion, opt, BATCH_SIZE, EPOCH, device)
    torch.save(model_trained, './model.pth')
    
    y, y_pred = model_trained.predict(dataloaders['test'], device=device)
    analyze.plot_heatmap(y, y_pred, list(cls2lbl.keys()))
    print()
    analyze.evaluation_report(y, y_pred)
    print()
    analyze.plot_loss_acc(history)


def initiate():

    parse = argparse.ArgumentParser()

    parse.add_argument('-e', '--epochs', dest='epochs', type=int, default=10)
    parse.add_argument('-b', '--batch',dest='batch', type=int, default=16)
    parse.add_argument('-r', '--resize',dest='img_resize', type=int, default=224)
    parse.add_argument('-d', '--device',dest='device', default=False, action='store_true')
    
    args = vars(parse.parse_args())
    main(args)


if __name__ == '__main__':
    initiate()