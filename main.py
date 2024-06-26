from Dataprocess import preprocessing, analyze
from Pipelines import img_aug_pipeline, data_pipeline
import model
from configs import BATCH_SIZE, device
import torch.nn as nn
import torch.optim as optim


def main():

    dfs, cls2lbl, lbl2cls = preprocessing.preprocess()
    data_transform = img_aug_pipeline.data_transforms
    
    analyze.fetch_examples(dfs['train'], lbl2cls, data_transform)
    dataloaders = data_pipeline.build_dataloader(dfs['train'], dfs['test'], data_transform)

    mdl = model.Model(3).to(device)
    opt = optim.Adam(mdl.parameters(), lr = 0.001)
    criterion = nn.CrossEntropyLoss()
    print()
    EPOCH = 10

    model_trained, history = mdl.fit(dataloaders, criterion, opt, BATCH_SIZE, EPOCH, device)

    y, y_pred = model_trained.predict(dataloaders['test'], device=device)
    analyze.plot_heatmap(y, y_pred, list(cls2lbl.keys()))
    print()
    analyze.evaluation_report(y, y_pred)
    print()
    analyze.plot_loss_acc(history)


if __name__ == '__main__':
    main()