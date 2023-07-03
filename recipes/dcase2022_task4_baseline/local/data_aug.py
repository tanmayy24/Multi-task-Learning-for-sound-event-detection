#Some codes are adopted from https://github.com/DCASE-REPO/DESED_task
import torch
import numpy as np
import random


def frame_shift(features, label=None, label_mtl=None, net_pooling=None):
    if label is not None:
        batch_size, _, _ = features.shape
        shifted_feature = []
        shifted_label = []
        shifted_label_mtl = []
        for idx in range(batch_size):
            shift = int(random.gauss(0, 90))
            shifted_feature.append(torch.roll(features[idx], shift, dims=-1))
            shift = -abs(shift) // net_pooling if shift < 0 else shift // net_pooling
            shifted_label.append(torch.roll(label[idx], shift, dims=-1))
            shifted_label_mtl.append(torch.roll(label_mtl[idx], shift, dims=-1))
        return torch.stack(shifted_feature), torch.stack(shifted_label), torch.stack(shifted_label_mtl)
    else:
        batch_size, _, _ = features.shape
        shifted_feature = []
        for idx in range(batch_size):
            shift = int(random.gauss(0, 90))
            shifted_feature.append(torch.roll(features[idx], shift, dims=-1))
        return torch.stack(shifted_feature)


def time_mask(features, labels=None, labels_mtl=None, net_pooling=None, mask_ratios=(10, 20)):
    if labels is not None:
        _, _, n_frame = labels.shape
        t_width = torch.randint(low=int(n_frame/mask_ratios[1]), high=int(n_frame/mask_ratios[0]), size=(1,))   # [low, high)
        t_low = torch.randint(low=0, high=n_frame-t_width[0], size=(1,))
        features[:, :, t_low * net_pooling:(t_low+t_width)*net_pooling] = 0
        labels[:, :, t_low:t_low+t_width] = 0
        labels_mtl[:, :, t_low:t_low+t_width] = 0
        return features, labels, labels_mtl
    else:
        _, _, n_frame = features.shape
        t_width = torch.randint(low=int(n_frame/mask_ratios[1]), high=int(n_frame/mask_ratios[0]), size=(1,))   # [low, high)
        t_low = torch.randint(low=0, high=n_frame-t_width[0], size=(1,))
        features[:, :, t_low:(t_low + t_width)] = 0
        return features


def feature_transformation(features, n_transform, choice, filter_db_range, filter_bands,
                           filter_minimum_bandwidth, filter_type, freq_mask_ratio, noise_snrs):
    if n_transform == 2:
        feature_list = []
        for _ in range(n_transform):
            features_temp = features
            if choice[0]:
                features_temp = filt_aug(features_temp, db_range=filter_db_range, n_band=filter_bands,
                                         min_bw=filter_minimum_bandwidth, filter_type=filter_type)
            if choice[1]:
                features_temp = freq_mask(features_temp, mask_ratio=freq_mask_ratio)
            if choice[2]:
                features_temp = add_noise(features_temp, snrs=noise_snrs)
            feature_list.append(features_temp)
        return feature_list
    elif n_transform == 1:
        if choice[0]:
            features = filt_aug(features, db_range=filter_db_range, n_band=filter_bands,
                                min_bw=filter_minimum_bandwidth, filter_type=filter_type)
        if choice[1]:
            features = freq_mask(features, mask_ratio=freq_mask_ratio)
        if choice[2]:
            features = add_noise(features, snrs=noise_snrs)
        return [features, features]
    else:
        return [features, features]


def filt_aug(features, db_range=[-6, 6], n_band=[3, 6], min_bw=6, filter_type="linear"):
    # this is updated FilterAugment algorithm used for ICASSP 2022
    if not isinstance(filter_type, str):
        if torch.rand(1).item() < filter_type:
            filter_type = "step"
            n_band = [2, 5]
            min_bw = 4
        else:
            filter_type = "linear"
            n_band = [3, 6]
            min_bw = 6

    batch_size, n_freq_bin, _ = features.shape
    n_freq_band = torch.randint(low=n_band[0], high=n_band[1], size=(1,)).item()   # [low, high)
    if n_freq_band > 1:
        while n_freq_bin - n_freq_band * min_bw + 1 < 0:
            min_bw -= 1
        band_bndry_freqs = torch.sort(torch.randint(0, n_freq_bin - n_freq_band * min_bw + 1,
                                                    (n_freq_band - 1,)))[0] + \
                           torch.arange(1, n_freq_band) * min_bw
        band_bndry_freqs = torch.cat((torch.tensor([0]), band_bndry_freqs, torch.tensor([n_freq_bin])))

        if filter_type == "step":
            band_factors = torch.rand((batch_size, n_freq_band)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            band_factors = 10 ** (band_factors / 20)

            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                freq_filt[:, band_bndry_freqs[i]:band_bndry_freqs[i + 1], :] = band_factors[:, i].unsqueeze(-1).unsqueeze(-1)

        elif filter_type == "linear":
            band_factors = torch.rand((batch_size, n_freq_band + 1)).to(features) * (db_range[1] - db_range[0]) + db_range[0]
            freq_filt = torch.ones((batch_size, n_freq_bin, 1)).to(features)
            for i in range(n_freq_band):
                for j in range(batch_size):
                    freq_filt[j, band_bndry_freqs[i]:band_bndry_freqs[i+1], :] = \
                        torch.linspace(band_factors[j, i], band_factors[j, i+1],
                                       band_bndry_freqs[i+1] - band_bndry_freqs[i]).unsqueeze(-1)
            freq_filt = 10 ** (freq_filt / 20)
        return features * freq_filt

    else:
        return features


def freq_mask(features, mask_ratio=16):
    batch_size, n_freq_bin, _ = features.shape
    max_mask = int(n_freq_bin/mask_ratio)
    if max_mask == 1:
        f_widths = torch.ones(batch_size)
    else:
        f_widths = torch.randint(low=1, high=max_mask, size=(batch_size,))   # [low, high)

    for i in range(batch_size):
        f_width = f_widths[i]
        f_low = torch.randint(low=0, high=n_freq_bin-f_width, size=(1,))

        features[i, f_low:f_low+f_width, :] = 0
    return features


def add_noise(features, snrs=(15, 30), dims=(1, 2)):
    if isinstance(snrs, (list, tuple)):
        snr = (snrs[0] - snrs[1]) * torch.rand((features.shape[0],), device=features.device).reshape(-1, 1, 1) + snrs[1]
    else:
        snr = snrs

    snr = 10 ** (snr / 20)
    sigma = torch.std(features, dim=dims, keepdim=True) / snr
    return features + torch.randn(features.shape, device=features.device) * sigma

