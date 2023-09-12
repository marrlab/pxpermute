from importlib import import_module

import click
import torch

from interpretation import run_interpretation_method, run_pxpermute
from util import load_model, prepare_test_data, read_config_file


@click.group()
def cli():
    pass


@cli.command()
@click.option("--output-path", default="../output", type=click.Path(exists=True))
@click.option(
    "--method",
    default="PxPermute",
    type=click.Choice(
        [
            "PxPermute",
            "DeepLift",
            "GuidedGradCam",
            "Saliency",
            "DeepLiftShap",
            "GradientShap",
            "InputXGradient",
            "IntegratedGradients",
            "GuidedBackprop",
            "Deconvolution",
            "Occlusion",
            "FeaturePermutation",
            "ShapleyValueSampling",
            "Lime",
            "KernelShap",
            "LRP",
        ]
    ),
)
def run_channel_importance(output_path, method="PxPermute"):
    # load parameters from config file
    parameters = read_config_file("config.cfg")
    if parameters["device"] == "cuda" and not torch.cuda.is_available():
        parameters["device"] = "cpu"
        click.echo("GPU is not available, using CPU instead")
    # load data and model
    metadata, loader, test_index, label_map, test_transform = prepare_test_data(
        **parameters
    )
    parameters["num_classes"] = len(label_map.keys())
    model = load_model(**parameters)
    # run interpretation method
    if method == "PxPermute":
        run_pxpermute(
            metadata,
            loader,
            model,
            output_path,
            test_index,
            test_transform,
            label_map,
            **parameters
        )
    else:
        mod = import_module("captum.attr")
        method = getattr(mod, method)
        ablator = method(model)
        run_interpretation_method(
            test_loader=loader, ablator=ablator, output_path=output_path, **parameters
        )


if __name__ == "__main__":
    run_channel_importance()
