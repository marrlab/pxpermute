# PXPermute: Unveiling staining importance in multichannel imaging flow cytometry

Imaging Flow Cytometry (IFC) allows rapid acquisition of numerous single-cell images per second, capturing information from multiple fluorescent channels. However, the traditional process of staining cells with fluorescently labeled conjugated antibodies for IFC analysis is time-consuming, expensive, and potentially harmful to cell viability. To streamline experimental workflows and reduce costs, it is crucial to identify the most relevant channels for downstream analysis. In this study, we introduce PXPermute, a user-friendly and powerful method for assessing the significance of IFC channels, particularly for cell profiling. Our approach evaluates channel importance by permuting pixel values within each channel and analyzing the resulting impact on machine learning or deep learning models. Through rigorous evaluation of three multi-channel IFC image datasets, we demonstrate PXPermute's potential in accurately identifying the most informative channels, aligning with established biological knowledge. To facilitate systematic investigations of channel importance and assist biologists in optimizing their experimental designs and finding the best biomarkers, we have released PXPermute as an easy-to-use open-source Python package.

## Documentation

For documentation, please refer to [python](python)#

## Contributing

We are happy about any contributions. Please send a pull request to the *develop* branch for any suggested changes.

## Citation

If you use pxpermute, please cite this paper:

```
@article {Boushehri2023.05.28.542646,
	author = {Sayedali Shetab Boushehri and Aleksandra Kornivetc and Domink J. E. Waibel and Salome Kazeminia and Fabian Schmich and Carsten Marr},
	title = {PXPermute: Unveiling staining importance in multichannel fluorescence microscopy},
	elocation-id = {2023.05.28.542646},
	year = {2023},
	doi = {10.1101/2023.05.28.542646},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/early/2023/05/30/2023.05.28.542646},
	eprint = {https://www.biorxiv.org/content/early/2023/05/30/2023.05.28.542646.full.pdf},
	journal = {bioRxiv}
}

```

