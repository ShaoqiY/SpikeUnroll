import os
import os.path as osp
import glob
import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model.SpkFlowNet import SpikeFlowNet


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class InputPadder:
    def __init__(self, dims, pad_size=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // pad_size) + 1) * pad_size - self.ht) % pad_size
        pad_wd = (((self.wd // pad_size) + 1) * pad_size - self.wd) % pad_size
        self._pad = [pad_wd // 2, pad_wd - pad_wd // 2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode="replicate") for x in inputs]

    def unpad(self, x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht - self._pad[3], self._pad[0], wd - self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


def load_spike_npy(path):
    spk = np.load(path)
    spk = np.unpackbits(spk, axis=2, bitorder="little").astype(np.float32)
    spk = torch.from_numpy(spk).unsqueeze(0)
    return spk


def load_rgb_image(path):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"读取图像失败: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    return img


def save_rgb_image(tensor, path):
    tensor = tensor.clamp(0, 1)
    img = (tensor * 255.0).byte().squeeze(0).permute(1, 2, 0).cpu().numpy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)


def load_checkpoint(model, ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"], strict=True)
    else:
        model.load_state_dict(ckpt, strict=True)
    print(f"Loaded: {ckpt_path}")


@torch.inference_mode()
def eval_real(model, device):
    dis_list_100 = []
    for i in range(20):
        H_list = []
        p_i = 2.5 + 5 * i
        for h in range(250):
            t_center = h * 95 / 249
            t_center += 2.5
            H_list.append(t_center - p_i)
        dis_list_100.append(H_list)

    dis_np_100 = []
    for H in dis_list_100:
        arr = np.array(H)[None, :, None]
        arr = np.repeat(arr, 400, axis=2)
        dis_np_100.append(arr)

    img_dir = "test/rs"
    spikers_dir = "test/spike_rs"
    spike_dir = "test/spike_gs"
    output_folder = "test/gs_flow"

    img_files = sorted(glob.glob(osp.join(img_dir, "*.png")))
    spk1_files = sorted(glob.glob(osp.join(spikers_dir, "*.npy")))
    spk2_files = sorted(glob.glob(osp.join(spike_dir, "*.npy")))

    os.makedirs(output_folder, exist_ok=True)

    num_frames = len(img_files)

    for i in range(num_frames):
        spk1_np = load_spike_npy(spk1_files[i])
        img1_np = load_rgb_image(img_files[i])

        spk1 = spk1_np.to(device, non_blocking=True)
        img1 = img1_np.to(device, non_blocking=True)

        img_padder = InputPadder(img1.shape, pad_size=16)
        img1_pad = img_padder.pad(img1)[0]

        for j in range(20):
            spk2_idx = i * 20 + j
            spk2 = load_spike_npy(spk2_files[spk2_idx]).to(device, non_blocking=True)

            spk_padder = InputPadder(spk1.shape, pad_size=16)

            distance = dis_np_100[j] / 100.0

            distance = torch.from_numpy(distance).float().unsqueeze(0).to(device, non_blocking=True)

            spk1_pad, spk2_pad, distance_pad = spk_padder.pad(spk1, spk2, distance)

            out = model(spk1_pad, spk2_pad, img1_pad, distance_pad)
            pred = img_padder.unpad(out["warp_img"])

            out_path = os.path.join(output_folder, f"frame{spk2_idx:04d}.png")
            save_rgb_image(pred, out_path)

    print(f"Finish!")
        


if __name__ == "__main__":
    model = SpikeFlowNet().to(device)
    ckpt_path = "./model_result/flow.pth"
    load_checkpoint(model, ckpt_path, device)

    eval_real(model=model, device=device)