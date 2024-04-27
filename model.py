import torch
import numpy as np
from tqdm.auto import tqdm
from collections import OrderedDict

import torch.nn as nn


class Model(nn.Module):
    def __init__(self, num_features):
        
        super(Model, self).__init__()
        
#         self.img_encoder = models.efficientnet_b3(weights='DEFAULT')
#         for param in self.img_encoder.parameters():
#             param.requires_grad = True
    
#         num_ftrs = self.img_encoder.classifier[1].in_features
#         self.img_encoder.classifier[1] = nn.Linear(num_ftrs, 4)

        self.layer1 = nn.Sequential(
                    OrderedDict([
                        ('conv1', nn.Conv2d(num_features, 128, 3, padding=1)),
                        ('relu1', nn.ReLU()),
                        ('batch1', nn.BatchNorm2d(128)),
                        ('maxpool1', nn.MaxPool2d(2, 2)),
                        ('dropout1', nn.Dropout2d(0.1)),
                        ('conv2', nn.Conv2d(128, 128, 3, padding=1)),
                        ('relu2', nn.ReLU()),
                        ('batch2', nn.BatchNorm2d(128)),
                        ('maxpool2', nn.MaxPool2d(2, 2)),
                        ('dropout2', nn.Dropout2d(0.1))
                    ])
                )
        
        self.layer2 =  nn.Sequential(
                    OrderedDict([
                        ('conv1', nn.Conv2d(128, 256, 3, padding=1)),
                        ('relu1', nn.ReLU()),
                        ('batch1', nn.BatchNorm2d(256)),
                        ('maxpool1', nn.MaxPool2d(2, 2)),
                        ('dropout1', nn.Dropout2d(0.1)),
                        ('conv2', nn.Conv2d(256, 256, 3, padding=1)),
                        ('relu2', nn.ReLU()),
                        ('batch2', nn.BatchNorm2d(256)),
                        ('maxpool2', nn.MaxPool2d(2, 2)),
                        ('dropout2', nn.Dropout2d(0.1))
                    ])
                )
        
        self.fc = nn.Sequential(
                    OrderedDict([
                        ('flat', nn.Flatten()),
                        ('linear1', nn.Linear(256 * 14 * 14, 256, True)),
                        ('batch1', nn.BatchNorm1d(256)),
                        ('dropout1', nn.Dropout1d(0.1)),
                        ('linear2', nn.Linear(256, 128, True)),
                        ('batch2', nn.BatchNorm1d(128)),
                        ('dropout2', nn.Dropout1d(0.1)),
                        ('output', nn.Linear(128, 7, True)),
                    ]))
    
    def fit(self, data, criterion, optimizer, batch_size, num_epochs=3, device=torch.device('cpu'), return_loss_acc=True):
    
        if return_loss_acc:
            tr_val_history = {'train': [], 'test': []}
            tr_val_acc_history = {'train': [], 'test': []}
        
        for epoch in range(num_epochs):
            print('-' * 50)
            print('Epoch {}/{}'.format(epoch+1, num_epochs))
            print('-' * 10)

    #         visualize_cnn_kernels(model, f'{epoch + 1:003}')
    #         visualize_cnn_featuremaps(model, sample_recs[2], f'{epoch + 1:003}', label_class)
            
            for phase in ['train', 'test']:
                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                running_loss = 0.0
                running_corrects = 0

                with tqdm(data[phase], unit='batch', position=0, leave=True) as pbar:
                    for img, lbl in pbar:

                        pbar.set_description(f"Epoch {epoch+1}")

                        img = img.to(device)
                        lbl = lbl.to(device)
                        outputs = self(img)
                        loss = criterion(outputs, lbl)

                        if phase == 'train':
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                        _, preds = (torch.max(outputs, 1))
                        running_loss += loss.item()
                        running_corrects += torch.sum(preds == lbl.data)
                        pbar.set_postfix(loss=loss.item() / batch_size, accuracy=torch.sum(preds == lbl.data).item() / batch_size)

                epoch_loss = running_loss / len(data[phase])
                epoch_acc = running_corrects.double() / len(data[phase])

                if return_loss_acc:
                    tr_val_history[phase].append(epoch_loss)
                    tr_val_acc_history[phase].append(epoch_acc.item())
                
                print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                            epoch_loss,
                                                            epoch_acc))
                
        if return_loss_acc:
            return self, (tr_val_history, tr_val_acc_history)
        return self
    
    
    def predict(self, data, device=torch.device('cpu')):
        y = []
        y_pred = []
        self.eval()
        
        with tqdm(data, unit='batch', position=0, leave=True) as pbar:
            for img, lbl in pbar:

                pbar.set_description(f"Evaluating")

                img = img.to(device)
                lbl = lbl.to(device)
                outputs = self(img)
                _, preds = torch.max(outputs, 1)
                y = y + [*np.array(lbl.cpu())]
                y_pred = y_pred + [*np.array(preds.cpu())]
        
        return y, y_pred
    
    
    def forward(self, x):
#         x = nn.functional.softmax(self.img_encoder(x), dim=1)
#         print(x.shape)
        x = self.layer1(x)
#         print(x.shape)
        x = self.layer2(x)
#         print(x.shape)
        x = self.fc(x)
#         print(x.shape)
        return x