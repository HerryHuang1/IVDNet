# IVDNet

VIDEO NOISE REMOVAL USING PROGRESSIVE DECOMPOSITIONWITH CONDITIONAL INVERTIBILITY

## Overview

This source code provides a PyTorch implementation of the IVDNet video denoising algorithm, as in 
H. Huang, Y. Quan, Y. Huang, J. Hu, Z. Lei. ["VIDEO NOISE REMOVAL USING PROGRESSIVE DECOMPOSITIONWITH CONDITIONAL INVERTIBILITY", arXiv preprint arXiv: xxx (2023).](https://arxiv.org/abs/xxx)


### Testing

If you want to denoise an image sequence using the pretrained model you can execute

```
test_ivdnet.py \
	--test_path <path_to_input_sequence> \
	--noise_sigma 30 \
	--save_path results
```

**NOTES**
* The image sequence should be stored under <path_to_input_sequence>
* The model has been trained for values of noise in [5, 55]
* run with *--no_gpu* to run on CPU instead of GPU
* run with *--save_noisy* to save noisy frames
* set *max_num_fr_per_seq* to set the max number of frames to load per sequence
* to denoise _clipped AWGN_ run with *--model_file model_clipped_noise.pth*
* run with *--help* to see details on all input parameters

### Training

DISCLAIMER: The weights shared in this repo were trained with a previous DALI version, v0.10.0, and pytorch v1.0.0. The training code was later updated to work with a more recent version of DALI. However, it has been reported that the perfomance obtained with this newer DALI version is not as good as the original one, see https://github.com/m-tassano/fastdvdnet/issues/51 for more details.

If you want to train your own models you can execute

```
train_ivdnet.py \
	--trainset_dir <path_to_input_mp4s> \
	--valset_dir <path_to_val_sequences> \
	--log_dir logs
```

**NOTES**
* As the dataloader in based on the DALI library, the training sequences must be provided as mp4 files, all under <path_to_input_mp4s>
* The validation sequences must be stored as image sequences in individual folders under <path_to_val_sequences>
* run with *--help* to see details on all input parameters

