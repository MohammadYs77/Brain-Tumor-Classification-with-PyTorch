import torch


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESIZE = 224
BATCH_SIZE = 16