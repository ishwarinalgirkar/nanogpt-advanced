import numpy as np
import torch

def get_batch(split, config):
    """
    Sample a random batch from the memory-mapped token file.
    """
    data = np.memmap(config['data_path'], dtype=np.uint16, mode='r')

    # train/val split by position in file
    n_val   = int(len(data) * config.get('val_split', 0.005))
    n_train = len(data) - n_val

    if split == 'train':
        data = data[:n_train]
    else:
        data = data[n_train:]

    B, T    = config['batch_size'], config['max_seq']

    # random starting positions
    ix = torch.randint(len(data) - T, (B,))

    # x is tokens at positions i..i+T-1
    # y is tokens at positions i+1..i+T
    x = torch.stack([torch.from_numpy(data[i:i+T].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+T+1].astype(np.int64)) for i in ix])

    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y
