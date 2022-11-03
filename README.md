[![DOI](https://zenodo.org/badge/475560916.svg)](https://zenodo.org/badge/latestdoi/475560916)

# Pipeline to detect, characterize and annotate WBC and RBC in peripheral blood slides (PBS)

All necessary Python packages are in requirements-pipeline.txt. This pipeline --- Haemorasis --- was developed and tested using Python 3.6.8 and on 8GB RAM + 12GB VRAM on CentOS Linux 8 (kernel: Linux 4.18.0-240.22.1.el8_3.x86_64). Haemorasis is composed of 7 steps (0 is optional) and is orchestrated by a snakemake pipeline (`Snakefile`):

0. The model checkpoints are downloaded if they are not available
1. Quality control of the PBS --- each 512*512 tile is quality controlled to filter out tiles with excessive/defficient cellular density and/or poor resolution (output stored as `{output.directory}/_quality_control/{slide.id}.h5`)
2. Segmentation of WBC and RBC --- segmentation coordinates are stored in hdf5 format, with a dataset for each cell (output stored as `{output.directory}/_segmented_wbc/{slide.id}.h5` and `{output.directory}/_segmented_rbc/{slide.id}.h5`)
3. Morphometric characterization of WBC --- morphometric features are stored in hdf5 format, with a dataset for each cell (output stored as `{output.directory}/_aggregates_wbc/{slide.id}.h5`)
4. Morphometric characterization of RBC --- morphometric features are stored in hdf5 format, with a dataset for each cell (output stored as `{output.directory}/_aggregates_rbc/{slide.id}.h5`)
5. Annotation of WBC in geojson format --- these annotations can be loaded into QuPath (output stored as `{output.directory}/_annotations_wbc/{slide.id}.h5`)
6. Annotation of RBC in geojson format --- these annotations can be loaded into QuPath (output stored as `{output.directory}/_annotations_rbc/{slide.id}.h5`)

Steps 3 and 4 are run in parallel, as well as steps 5 and 6. In the examples above, `{slide.id}` is the `basename` of the slide until the first point (if the slide path is `/homes/user/slide_32.0.1.tiff` then `{slide.id}` is `slide_32`). The output directory is specified in the scripts below as `-o`.

## Install instructions

To test the code provided here, you must first have a working Python 3.6.8 version, at least 8GB of RAM and a GPU card with at least 12GB of VRAM and CUDA capabilities. This has been tested on operating systems based on either Ubuntu or CentOS. Then:

1. Install the packages listed in `requirements-pipeline.txt`
2. Download a peripheral blood slide from https://www.ebi.ac.uk/biostudies/studies/S-BIAD440, a BioImage Archive dataset containing all slides used in this work 
3. Run the Haemorasis described below ("Usage").

## Docker container and further instructions

A Docker container has been made available to facilitate the application of this pipeline in https://hub.docker.com/repository/docker/josegcpa/blood-cell-detection. Instructions on Docker usage are provided in the Docker container landing page and in the Supplementary Materials of the publication where this software was used (https://www.medrxiv.org/content/10.1101/2022.04.19.22273757v1) under "Setting up Haemorasis" and "Running Haemorasis".

### Usage of `run-slide.sh` (to run a single digitalised slide):

```
arguments (required):
                -i : input slide
                -o : output directory

arguments (optional):
                -q : path to quality control network checkpoint (if not in -q will be downloaded)
                -u : path to WBC segmentation network checkpoint (if not in -u will be downloaded)
                -x : path to RBC object filtering XGboost model parameters (if not in -u will be downloaded)
                -r : rescale factor (default = 1.0)
                -m : mode (local or cluster (only LSF supported))

example:
                bash run-slide.sh -i slide_dir -o output_dir -f tiff -r 1.1
```

An example of running Haemorasis is, for a slide `/homes/user/slides/slide_a.tiff` and the output directory in `/homes/user/output` is `sh run-slide.sh -i /homes/user/slides/slide_a.tiff -o /homes/user/output`. `sh run-slide.sh -h` displays other available options.

## Usage

### Usage of `run-folder.sh` (to run a folder of digitalised slides with a given format):

```
arguments (required):
                -i : input directory
                -f : file extension (i.e. ndpi, svs, tiff, etc...)
                -o : output directory

arguments (optional):
                -q : path to quality control network checkpoint (if not in -q will be downloaded)
                -u : path to WBC segmentation network checkpoint (if not in -u will be downloaded)
                -x : path to RBC object filtering XGboost model parameters (if not in -u will be downloaded)
                -r : rescale factor (default = 1.0)
                -m : mode (local or cluster (only LSF supported))

example:
                bash run-folder.sh -i slide_dir -o output_dir -f tiff -r 1.1
```
