import os
import sys
import urllib.request
import zipfile
from glob import glob
import ssl
import certifi
import torch as t
from PIL import Image
from torch.utils.data import Dataset, random_split

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
sys.path.append(grandparent_dir)

class NotoEmoji(Dataset):
    url = "https://www.dropbox.com/s/y6tlfrg0p634csj/noto-emoji-128.zip?dl=1"

    def __init__(self, data_dir, transform=lambda x: x):
        noto_dir = os.path.join(data_dir, 'noto-emoji-128')
        if not os.path.exists(noto_dir):
            noto_zip = os.path.join(data_dir, 'noto-emoji-128.zip')
            ssl_context = ssl.create_default_context(cafile=certifi.where())
            with urllib.request.urlopen(self.url, context=ssl_context) as r:
                with open(noto_zip, 'wb') as f:
                    f.write(r.read())
            with zipfile.ZipFile(noto_zip, 'r') as zip_ref:
                zip_ref.extractall(noto_dir)

        self.samples = [(transform(Image.open(f)), 0) for f in glob(os.path.join(noto_dir, '128/*.png'))]

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def train_val_split(self, percent_val=0.2, seed=0):
        n_images = len(self)
        n_val = int(n_images * percent_val)
        return random_split(self, [n_images - n_val, n_val], generator=t.Generator().manual_seed(seed))

if __name__ == '__main__':
    NotoEmoji('/tmp')