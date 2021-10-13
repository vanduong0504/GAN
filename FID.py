import math
import torch
import glob
import argparse
import numpy as np
import torch.nn as nn
from PIL import Image
from scipy.linalg import sqrtm
from torchvision.transforms import transforms
from torchvision.models import inception_v3
from torchvision.models.utils import load_state_dict_from_url

def option():
    agr = argparse.ArgumentParser(description="Calculate the FID score")
    agr.add_argument("--real_path", required=True, type=str, metavar="RP",
                            help="path to real images")
    agr.add_argument("--fake_path", required=True, type=str, metavar="FP",
                            help="path to fake images")
    agr.add_argument("--device", default="cuda", type=str,
                        help="Set gpu mode: [cpu, cuda].")
    return agr.parse_args()


def preprocess(image):
    tensor = nn.functional.interpolate(image, size=(299, 299), mode='bilinear', align_corners=False)
    tensor = 2 * tensor - 1  # Scale from range (0, 1) to range (-1, 1)
    return tensor

def get_actiavtion(path, model, device):
    activation = []
    trans = transforms.ToTensor()
    for extension in (".jpg", ".png"):
        for image in glob.glob(path + f"/*{extension}"):
            image = Image.open(image).convert('RGB')
            tensor = trans(image).unsqueeze(0).to(device)
            tensor = preprocess(tensor)
            img_tensor = model(tensor).detach().cpu().squeeze(0).squeeze(1).squeeze(1)
            activation.append(img_tensor)
    return torch.stack(activation)

def Inception_V3():
    """
    This function define inception_v3 model.
    """
    FID_WEIGHTS_URL = 'https://github.com/mseitzer/pytorch-fid/releases/download/fid_weights/pt_inception-2015-12-05-6726825d.pth'
    
    inception = inception_v3(num_class = 1008, aux_logits = False)
    state_dict = load_state_dict_from_url(FID_WEIGHTS_URL, progress=True)
    inception.load_state_dict(state_dict)

    block0 = [
        inception.Conv2d_1a_3x3,
        inception.Conv2d_2a_3x3,
        inception.Conv2d_2b_3x3,
        nn.MaxPool2d(kernel_size=3, stride=2)]

    block1 = [
        inception.Conv2d_3b_1x1,
        inception.Conv2d_4a_3x3,
        nn.MaxPool2d(kernel_size=3, stride=2)]

    block2 = [
        inception.Mixed_5b,
        inception.Mixed_5c,
        inception.Mixed_5d,
        inception.Mixed_6a,
        inception.Mixed_6b,
        inception.Mixed_6c,
        inception.Mixed_6d,
        inception.Mixed_6e]

    block3 = [
        inception.Mixed_7a,
        inception.Mixed_7b,
        inception.Mixed_7c,
        nn.AdaptiveAvgPool2d(output_size=(1, 1))]

    return nn.Sequential(*block0, *block1,*block2, *block3)

def FID(real, fake):
    mu_r, sig_r = torch.mean(real, 0), torch.Tensor(np.cov(real.cpu().numpy(), rowvar=False))
    mu_f, sig_f = torch.mean(fake, 0), torch.Tensor(np.cov(fake.cpu().numpy(), rowvar=False))
    
    matrix_sqrt = (sig_r@sig_f).cpu().numpy()
    matrix_sqrt = torch.Tensor(sqrtm(matrix_sqrt).real)

    mu_dif = torch.matmul(mu_r - mu_f, mu_r - mu_f)
    sig_dif = torch.trace(sig_r + sig_f - 2*matrix_sqrt)
    return (mu_dif + sig_dif).detach().item()

if __name__ == "__main__":

    parser = option()

    inception = Inception_V3().to(parser.device)
    inception.eval()

    with torch.no_grad():
        real = get_actiavtion(parser.real_path, inception, parser.device).to(parser.device)
        fake = get_actiavtion(parser.fake_path, inception, parser.device).to(parser.device)

        fid = FID(real, fake)
        if math.isclose(fid,0) or fid<0:
            print("FID: 0")
        else: 
            print(f"FID: {fid:.2f}")