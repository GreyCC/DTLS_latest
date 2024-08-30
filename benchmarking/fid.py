import argparse
import torch
import numpy as np

from scipy import linalg
from tqdm import tqdm
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader
from calc_inception import load_patched_inception_v3
from pathlib import Path
from PIL import Image


@torch.no_grad()
def extract_features(loader, inception, device):
    pbar = tqdm(loader)

    feature_list = []

    for img in pbar:
        img = img.to(device)
        feature = inception(img)[0].view(img.shape[0], -1)
        feature_list.append(feature.to('cpu'))

    features = feature_list.pop(0)
    for feature in feature_list:
        features = torch.cat((features, feature),dim=0)

    return features


def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid

class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


if __name__ == '__main__':
    device = 'cuda:1'

    parser = argparse.ArgumentParser()

    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--path_a', type=str, default="/hdda/Datasets/celeba/data1024x1024")
    parser.add_argument('--path_b', type=str, default="/home/user/Desktop/Code/DTLS_Latest_update/eval/32_1024_test_on_celeba_even/result_only")

    args = parser.parse_args()

    inception = load_patched_inception_v3().eval().to(device)

    transform = transforms.Compose(
        [
            transforms.Resize( (args.size, args.size) ),
            #transforms.RandomHorizontalFlip(p=0.5 if args.flip else 0),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dset_a = Dataset(args.path_a, args.size)
    loader_a = DataLoader(dset_a, batch_size=args.batch, num_workers=4)

    features_a = extract_features(loader_a, inception, device).numpy()
    print(f'extracted {features_a.shape[0]} features')

    real_mean = np.mean(features_a, 0)
    real_cov = np.cov(features_a, rowvar=False)
    
    #for folder in os.listdir(args.path_b):
    # for folder in range(args.iter,args.end+1):
    #     folder = 'eval_%d'%(folder*10000)
    #     if os.path.exists(os.path.join( args.path_b, folder )):
    #         print(folder)

    dset_b = Dataset(args.path_b, args.size)
    loader_b = DataLoader(dset_b, batch_size=args.batch, num_workers=4)

    features_b = extract_features(loader_b, inception, device).numpy()
    print(f'extracted {features_b.shape[0]} features')

    sample_mean = np.mean(features_b, 0)
    sample_cov = np.cov(features_b, rowvar=False)

    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)

    print(' fid:', fid)
