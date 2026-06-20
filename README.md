# NuMorph 3DUnet

The package performs cell nuclei segmentation on large light-sheet imaging dataset. The models that the package uses can be found [here](https://bitbucket.org/steinlabunc/numorph/downloads/) for download. 

This package containes the original 3DUnet used in the NuMorph pipeline. A detailed describtion of the architecture and training procedure can be found in the [publication](https://doi.org/10.1016/j.celrep.2021.109802).


>The PyPi package is intended to be installed in a Nvidia optimized [container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow/tags?version=20.10-tf2-py3) with Tensorflow and used as a nextflow module in the pipeline [nf-core/lsmquant](https://github.com/nf-core/lsmquant). The container is hosted by the nf-core community repository on quay.io .   

## Installation
The package can also be used within a conda environment (not recommended). 

Clone the repository to your workstation.
The `numorphunet.yml` defines the necessary dependencies for running the prediction. You need to have `conda` installed to create the environment with the following command: 
```
conda env create -f numorphunet.yml
```
Activate the environment with: 
```
conda activate 3dunet
```
Install the numorph 3DUnet in the `3dunet`conda env by using the following command in the directory of the `pyproject.toml` file :
```
pip install .
```

## Usage

Once installed, you can run the cell segmentation tool using the command:

```
numorph_3dunet.predict -i /path/to/input/directory -o /path/to/output/directory --n_channels 1 --sample_name TEST1 --model /path/to/model_file.h5
```

Required arguments:
- `-i`: Input image directory
- `-o`: Output directory (will be created if it doesn't exist)
- `--n_channels`: Number of channels
- `--sample_name`: Sample name for output files
- `--model`: Model file (.h5)

Optional arguments:
- `-g`: GPU tag (default: 0)
- `--pred_threshold`: Prediction threshold (default: 0.5)
- `--int_threshold`: Minimum intensity threshold (default: 200)
- `--overlap`: Overlap between chunks [x y z] (default: 16 16 8)

See full help with `numorph_3dunet.predict --help`


# Credits

The pip package was originally developed by Carolin Schwitalla and contains the original work of Oleh Krupa who is the main developer of the 3DUnet and corresponding models used by the NuMorph toolbox.
> **NuMorph: Tools for cortical cellular phenotyping in tissue-cleared whole-brain images**
>
> Krupa O, Fragola G, Hadden-Ford E, Mory JT, Liu T, Humphrey Z, Rees BW, Krishnamurthy A, Snider WD, Zylka MJ, Wu G, Xing L, Stein JL.
>
> Cell Rep. 2021 Oct 12, doi: [10.1016/j.celrep.2021.109802](https://doi.org/10.1016%2Fj.celrep.2021.109802)