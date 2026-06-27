import os
import os.path as osp
os.environ['TORCH_HOME'] = '/opt/data/private/shaoqi/.cache/torch'
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import glob
from model.SURFNet import SpikeImageFusion


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class FixedPadder:
    def __init__(self, dims, target_size):
        self.ht, self.wd = dims[-2:]
        self.tgt_ht, self.tgt_wd = target_size

        pad_ht = max(0, self.tgt_ht - self.ht)
        pad_wd = max(0, self.tgt_wd - self.wd)

        self._pad = [
            pad_wd // 2,
            pad_wd - pad_wd // 2,
            0,
            pad_ht,
        ]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        c = [
            self._pad[2],
            self._pad[2] + self.ht,
            self._pad[0],
            self._pad[0] + self.wd,
        ]
        return x[..., c[0]:c[1], c[2]:c[3]]


def eval_real(model, device):

    model.eval()

    img_dir = 'test/gs_flow'
    spike_dir = 'test/spike_gs'
    output_folder = 'test/gs'

    model.reset_state()

    img_files = sorted(glob.glob(osp.join(img_dir, '*.png')))
    spike_files = sorted(glob.glob(osp.join(spike_dir, '*.npy')))

    os.makedirs(output_folder, exist_ok=True)

    img_padder = None
    spike_padder = None

    num_frames = len(img_files)

    for i in range(num_frames):
        if i % 20 == 0:
            model.reset_state()

        spike_arr = np.load(spike_files[i])
        spike_arr = np.unpackbits(
            spike_arr, axis=2, bitorder="little"
        ).astype(np.float32)

        spike_0 = torch.from_numpy(spike_arr).unsqueeze(0).to(device)

        prev = cv2.imread(img_files[i], cv2.IMREAD_COLOR)
        prev = cv2.cvtColor(prev, cv2.COLOR_BGR2RGB)
        img_0 = torch.from_numpy(prev).permute(2, 0, 1).unsqueeze(0).to(device)
        img_0 = img_0.float() / 255.0

        if img_padder is None:
            _, _, h1, w1 = img_0.shape
            _, _, h2, w2 = spike_0.shape

            img_padder = FixedPadder((h1, w1), target_size=(504, 800))
            spike_padder = FixedPadder((h2, w2), target_size=(252, 400))

        img_0 = img_padder.pad(img_0)
        spike_0 = spike_padder.pad(spike_0)

        with torch.inference_mode():
            output = model(img_0[0], spike_0[0])

        output = output.clamp(0.0, 1.0)
        output = img_padder.unpad(output)

        out_img = (output * 255.0).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
        out_bgr = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)

        out_path = os.path.join(output_folder, f"frame{i:04d}.png")
        cv2.imwrite(out_path, out_bgr)

    print("Finish!")

        


if __name__ == '__main__': 
    model = SpikeImageFusion() 
    checkpoint = torch.load('./model_result/SURF-Net.pth', map_location=device) 
    model.load_state_dict(checkpoint) 
    model = model.to(device)
    with torch.no_grad(): 
        eval_real(model=model, device=device)