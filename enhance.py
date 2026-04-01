import torch
from PIL import Image
import torchvision.transforms as transforms
from model import networks
from options.base_options import TrainOpt
import numpy as np
from collections import OrderedDict
from util.util import tensor2im, atten2im
from skimage.restoration import denoise_nl_means

opt = TrainOpt()

netG = networks.define_G(input_nc=3, output_nc=3, ngf=64, which_model_netG='sid_unet_resize', norm='instance', skip=True, opt=opt)

state_dict = torch.load('model/200_net_G_A.pth', map_location='cpu')

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k.startswith('module.'):
        k = k.replace('module.', '', 1)
    new_state_dict[k] = v

netG.load_state_dict(new_state_dict)
netG.eval()

def enhance_image(input_image):
    input_image = input_image.convert('RGB')
    A_size = input_image.size
    A_size = (A_size[0] // 16 * 16, A_size[1] // 16 * 16)
    input_image = input_image.resize(A_size, Image.Resampling.BICUBIC)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
                             (0.5, 0.5, 0.5))
    ])

    input_tensor = transform(input_image).unsqueeze(0)

    r = input_tensor[0, 0, :, :] + 1
    g = input_tensor[0, 1, :, :] + 1
    b = input_tensor[0, 2, :, :] + 1
    A_gray = 1. - (0.299*r+0.587*g+0.114*b)/2.
    A_gray = torch.unsqueeze(A_gray, 0)
    A_gray = torch.unsqueeze(A_gray, 0)

    real_A = input_tensor

    with torch.no_grad():
        fake_B, latent_real_A = netG(real_A, A_gray)

    fake_B = Image.fromarray(tensor2im(fake_B.data))
    A_gray = Image.fromarray(atten2im(A_gray.data))

    return fake_B

def denoise_image(image):
    image_np = np.array(image)

    channel_axis = -1
    denoised_channels = []
    for c in range(image_np.shape[2]):
        denoised_channel = denoise_nl_means(
            image_np[:, :, c],
            h=0.08,
            fast_mode=True,
            patch_size=2,
            patch_distance=3,
            channel_axis=None
        )
        denoised_channels.append(denoised_channel)
    denoised_img = np.stack(denoised_channels, axis=2)

    denoised_image = Image.fromarray((denoised_img * 255).astype(np.uint8))
    return denoised_image