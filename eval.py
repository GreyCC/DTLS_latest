import torch
import torch.nn.functional as F
import os
import numpy as np
import argparse

from tqdm import tqdm
from torch.utils import data
from models import Encoder, Decoder
from operation import InfiniteSamplerWrapper
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision import utils as vutils
from PIL import Image
from pathlib import Path

def downgrade_in_seq(image, target_size, domain_list):
    ori_size = image.shape[2]

    for j in range(image.shape[0]):
        image_a = image[j].clone().unsqueeze(0)
        for i in range(int(target_size[j])):  # target_size = t
            image_a = F.interpolate(image_a, size=domain_list[i], mode='bicubic', antialias=True)
        image[j] = F.interpolate(image_a, size=ori_size, mode='bicubic', antialias=True).squeeze()
    return image


class Dataset(data.Dataset):
    def __init__(self, folder, image_size, exts=['jpg', 'jpeg', 'png']):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = [p for ext in exts for p in Path(f'{folder}').glob(f'**/*.{ext}')]

        self.transform = transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1)
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        img = Image.open(path)
        return self.transform(img)


def inference(args, domains_interval):
    data_root = args.path
    checkpoint = args.ckpt
    batch_size = args.batch_size
    im_size = args.im_size
    use_cuda = True
    dataloader_workers = args.workers
    saved_image_folder = f"eval/{args.output_path}"
    if not os.path.exists(saved_image_folder):
        os.makedirs(saved_image_folder)
        os.makedirs(f"{saved_image_folder}/with_gt/")
        os.makedirs(f"{saved_image_folder}/result_only/")

    num_domains = len(domains_interval)

    if use_cuda:
        device = f"cuda:{args.cuda}"
    else:
        device = torch.device("cpu")

    transform_list = [
        transforms.Resize((int(im_size), int(im_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    trans = transforms.Compose(transform_list)

    if 'lmdb' in data_root:
        from operation import MultiResolutionDataset
        dataset = MultiResolutionDataset(data_root, trans, 1024)
    else:
        dataset = Dataset(data_root, im_size)

    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                 sampler=InfiniteSamplerWrapper(dataset), num_workers=dataloader_workers,
                                 pin_memory=False))
    '''
    loader = MultiEpochsDataLoader(dataset, batch_size=batch_size, 
                               shuffle=True, num_workers=dataloader_workers, 
                               pin_memory=True)
    dataloader = CudaDataLoader(loader, 'cuda')
    '''

    encoder = Encoder()
    decoder = Decoder()

    if checkpoint != 'None':
        ckpt = torch.load(checkpoint,map_location=device)
        encoder.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['enc'].items()})
        decoder.load_state_dict({k.replace('module.', ''): v for k, v in ckpt['dec'].items()})
        del ckpt

    encoder.to(device)
    decoder.to(device)

    for n in tqdm(range(args.samples)):

        gt_img = next(dataloader).to(device)
        target_domain = np.ones(gt_img.shape[0]) * (num_domains - 1)
        lr_img = downgrade_in_seq(gt_img.clone(), target_domain, domains_interval)

        with torch.no_grad():
            sr_img = lr_img.clone()
            previous_prediction = None
            inverse_momentum = 0
            for i in reversed(range(num_domains - 1)):
                t = torch.ones(args.batch_size, device=device).long() * (i + 1)
                if previous_prediction is None:
                    inverse_momentum_LR = 0
                else:
                    inverse_momentum_LR = downgrade_in_seq(inverse_momentum.clone(), t, domains_interval)

                l, h = encoder(sr_img + inverse_momentum_LR, t)
                result = decoder(l, t, h)
                temp_result = result.clone()
                if previous_prediction is None:
                    previous_prediction = temp_result.clone()
                inverse_momentum += (previous_prediction - temp_result)
                previous_prediction = temp_result.clone()

                sr_img = downgrade_in_seq(result, t - 1, domains_interval)

        # vutils.save_image(torch.cat([
        #     lr_img.add(1).mul(0.5),
        #     result.add(1).mul(0.5),
        #     gt_img.add(1).mul(0.5)]),
        #     saved_image_folder + '/with_gt/%d.jpg' % n)

        vutils.save_image(result.add(1).mul(0.5),
                          saved_image_folder + '/result_only/%d.jpg' % n)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DTLS')

    parser.add_argument('--path', type=str, default='/hdda/Datasets/celeba/data1024x1024',
                        help='path of resource dataset, should be a folder that has one or many sub image folders inside')
    parser.add_argument('--output_path', type=str, default='32_1024_test_on_celeba_even', help='Output path for the train results')
    parser.add_argument('--cuda', type=int, default=1, help='index of gpu to use')
    # parser.add_argument('--name', type=str, default='32_1024_test_on_celeba_uneven', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=1, help='mini batch number of images')
    parser.add_argument('--im_size', type=int, default=1024, help='image resolution')
    parser.add_argument('--samples', type=int, default=2000, help='number of samples to be SR')

    parser.add_argument('--ckpt', type=str, default="150000_32_1024_even.pth") #'train_results/'
                                                    # 'DTLS_super_resolution_32_1024_uneven_iii/models/200000.pth',
                                                    #     help='checkpoint weight path if have one')

    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')

    ### Args for DTLS ###
    # domains_interval =[512, 448, 384, 320, 256, 64, 16] #512
    # domains_interval =[256, 208, 160, 112, 64, 32, 16] # 256
    domains_interval = [1024, 896, 768, 512, 256, 128, 32]  # 1024

    args = parser.parse_args()

    inference(args, domains_interval)
