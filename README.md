# SpikeUnroll

Code and dataset for Chromatic HFR and HDR Video via Virtual Spike Frame Unrolling

# Datasets

## Public Dataset

We use the dataset provided in the following paper:

[EvUnroll: Neuromorphic Events based Rolling Shutter Image Correction](https://github.com/zxyemo/EvUnroll?tab=readme-ov-file)

## Our Dataset

We also provide our collected dataset for evaluation.
Baidu Netdisk：https://pan.baidu.com/xxxx  

# How to Run Demo
Create and activate the environment:

```bash
conda create -n spike_env python=3.9
conda activate spike_env

pip install -r requirements.txt
```
Run the demo code:
```bash
python inference_flow.py # Generate spike-flow-unrolling results.
pyhton inference.py      # Generate the refined results using the SURF-Net network.
```
