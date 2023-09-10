import os.path
from time import process_time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scifAI.dl.custom_transforms import ShuffleChannel
from scifAI.dl.dataset import DatasetGenerator
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from torchvision import transforms


def run_interpretation_method(test_loader, ablator, output_path, require_baseline=False, require_sliding_window=False,
                              **kwargs):
    heatmaps = torch.empty(0, dtype=torch.float32, device=kwargs["device"])
    t1_start = process_time()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(kwargs["device"]).float(), data[1].to(kwargs["device"]).reshape(-1).long()
            if require_baseline:
                baselines = torch.zeros(inputs.shape).to(kwargs["device"])
                attr = ablator.attribute(inputs, target=labels, baselines=baselines)
            elif require_sliding_window:
                attr = ablator.attribute(inputs, target=labels, sliding_window_shapes=kwargs["sliding_window_shapes"])
            else:
                attr = ablator.attribute(inputs, target=labels)
            heatmaps = torch.cat((heatmaps, torch.from_numpy(
                np.percentile(torch.flatten(attr, start_dim=-2).cpu().numpy(), q=50, axis=-1)).to(kwargs["device"])))
    heatmaps_mean = torch.mean(heatmaps, dim=0)
    plt.bar(kwargs["channel_names"], heatmaps_mean.cpu(), color='grey')
    plt.savefig(os.path.join(output_path, str(ablator.get_name()) + ".png"))

    t1_stop = process_time()
    print("Elapsed time:", t1_stop, t1_start)

    print("Elapsed time during the whole program in seconds:",
          t1_stop - t1_start)
    return heatmaps_mean


def run_pxpermute(metadata, test_loader, model, output_path, test_index, test_transform, label_map, selected_channels,
                  scaling_factor, reshape_size, device, num_classes, num_channels, batch_size=128, num_workers=4, **kwargs):
    class_names = [c for c in label_map.keys()]
    t1_start = process_time()
    y_true = list()
    y_pred = list()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data[0].to(device).float(), data[1].to(device).long()
            outputs = model(inputs)
            pred = outputs.argmax(dim=1)
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(pred)):
                y_true.append(labels[i].item())
                y_pred.append(pred[i].item())
    f1_score_original = f1_score(y_true, y_pred, average=None, labels=np.arange(num_classes))
    min_mean_dif = 1.0
    # candidate = 0
    df_all = pd.DataFrame([], columns=class_names)
    for c in range(num_channels):
        f1_score_diff_from_original_per_channel_per_shuffle = []
        transform = test_transform.copy()
        transform.append(ShuffleChannel(channels_to_shuffle=[c]))
        for s in range(kwargs["shuffle_times"]):
            dataset = DatasetGenerator(metadata=metadata.loc[test_index, :],
                                       label_map=label_map,
                                       selected_channels=selected_channels,
                                       scaling_factor=scaling_factor,
                                       reshape_size=reshape_size,
                                       transform=transforms.Compose(transform))
            dataloader = DataLoader(dataset,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    num_workers=num_workers)
            y_true = list()
            y_pred = list()
            with torch.no_grad():
                for data in dataloader:
                    inputs, labels = data[0].to(device).float(), data[1].to(device).reshape(-1).long()
                    outputs = model(inputs)
                    pred = outputs.argmax(dim=1)
                    for i in range(len(pred)):
                        y_true.append(labels[i].item())
                        y_pred.append(pred[i].item())
                f1_score_per_channel = f1_score(y_true, y_pred, average=None, labels=np.arange(num_classes))
                f1_score_diff_from_original_per_channel_per_shuffle.append(f1_score_original - f1_score_per_channel)
        mean_along_columns = np.mean(f1_score_diff_from_original_per_channel_per_shuffle, axis=0)
        mean_dif = np.mean(mean_along_columns)
        if mean_dif < min_mean_dif and mean_dif > 0 and not selected_channels[c]:
            min_mean_dif = mean_dif
            # candidate = selected_channels[c]
        df_diff = pd.DataFrame(np.atleast_2d(f1_score_diff_from_original_per_channel_per_shuffle),
                               columns=class_names)
        df_mean_diff = pd.DataFrame(np.atleast_2d(mean_along_columns), columns=class_names)
        df_all = pd.concat([df_all, df_mean_diff], ignore_index=True, sort=False)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax = df_diff.boxplot()
        ax.set_xticklabels(class_names, rotation=45)
        fig.savefig(os.path.join(output_path, f"pxpermute-{selected_channels[c]}.png"))
    plt.bar(np.asarray(kwargs["channel_names"])[selected_channels], df_all.T.mean(), color='Grey')
    plt.savefig(os.path.join(output_path, "pixel-permutation-method-final.png"))
    t1_stop = process_time()
    print("Elapsed time:", t1_stop, t1_start)

    print("Elapsed time during the whole program in seconds:",
          t1_stop - t1_start)
