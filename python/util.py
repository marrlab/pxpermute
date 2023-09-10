import ast
import configparser

import numpy as np
import scifAI
import torch
from scifAI.dl.dataset import DatasetGenerator
from scifAI.dl.models import PretrainedModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import transforms

from transforms import MinMaxScaler


def read_config_file(config_file_path):
    config = configparser.ConfigParser()
    config.read(config_file_path)
    parameters = {}

    # General section
    general_section = config['general']
    parameters['batch_size'] = int(general_section.get('batch_size'))
    parameters['num_workers'] = int(general_section.get('num_workers'))
    parameters['device'] = general_section.get('device').strip("'")  # Remove single quotes

    # param.data section
    data_section = config['param.data']
    parameters['data_path'] = data_section.get('data_path').strip("'")  # Remove single quotes
    # Check if stats is present (optional)
    if 'stats' in data_section:
        parameters['stats'] = ast.literal_eval(data_section.get('stats'))
    parameters['scaling_factor'] = float(data_section.get('scaling_factor'))
    parameters['reshape_size'] = int(data_section.get('reshape_size'))
    parameters['selected_channels'] = ast.literal_eval(data_section.get('selected_channels'))
    parameters['channel_names'] = ast.literal_eval(data_section.get('channel_names'))
    parameters['num_channels'] = len(parameters['selected_channels'])
    assert parameters['num_channels'] == len(parameters['channel_names']), 'the legth of selected_channels does not ' \
                                                                           'match with channel_names '

    # param.model section
    model_section = config['param.model']
    parameters['model_path'] = model_section.get('model_path').strip("'")  # Remove single quotes

    # param.interpretation section
    interpretation_section = config['param.interpretation']
    parameters['shuffle_times'] = int(interpretation_section.get('shuffle_times'))
    parameters['require_baseline'] = interpretation_section.getboolean('require_baseline', fallback=False)
    parameters['require_sliding_window'] = interpretation_section.getboolean('require_sliding_window', fallback=False)

    # Check for sliding_window_shapes (optional)
    if 'sliding_window_shapes' in data_section:
        parameters['sliding_window_shapes'] = ast.literal_eval(interpretation_section.get('sliding_window_shapes'))

    return parameters


def load_model(model_path, num_classes, num_channels, device, **kwargs):
    """
    Load model from model_path
    :param model_path:
    :return: model
    """
    model = PretrainedModel(num_classes, num_channels, pretrained=False)

    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model


def prepare_test_data(data_path, test_index=[], selected_channels=[], reshape_size=64, scaling_factor=255.,
                      batch_size=128, num_workers=4, stats=None, seed_value=42, **kwargs):
    """
    Prepare test data for interpretation
    :param data_path:
    :param test_index:
    :param label_map:
    :param selected_channels:
    :param scaling_factor:
    :param reshape_size:
    :param test_transform:
    :param batch_size:
    :param num_workers:
    :return: testloader, test_index, label_map, test_transform
    """
    # load metadata and get test index
    metadata = scifAI.metadata_generator(data_path)
    row_index = metadata.label != "unknown"
    metadata = metadata.loc[row_index, :].reset_index(drop=True)
    if len(test_index) == 0:
        index_to_split = metadata.index.tolist()
        try:
            _, test_index, _, _ = train_test_split(index_to_split,
                                                   metadata.loc[index_to_split, "label"].index.tolist(),
                                                   stratify=metadata.loc[index_to_split, "label"].tolist(),
                                                   test_size=0.2,
                                                   random_state=seed_value)
        except ValueError:
            # take all data as test data if stratification is not possible
            test_index = index_to_split

    # get label map and num_classes
    label_map = dict(zip(sorted(set(metadata.loc[test_index, "label"])),
                         np.arange(len(set(metadata.loc[test_index, "label"])))))

    test_transform = [] if stats is None else MinMaxScaler(min_in=stats["lower_bound"],
                                                           max_in=stats["upper_bound"],
                                                           min_out=0.,
                                                           max_out=1.)
    # prepare test data
    dataset = DatasetGenerator(metadata=metadata.loc[test_index, :],
                               label_map=label_map,
                               selected_channels=selected_channels,
                               scaling_factor=scaling_factor,
                               reshape_size=reshape_size,
                               transform=transforms.Compose(test_transform))

    test_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers)
    return metadata, test_loader, test_index, label_map, test_transform
