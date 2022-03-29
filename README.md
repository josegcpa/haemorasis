# Pipeline to detect, characterize and annotate WBC and RBC in whole blood slides (WBS)

All necessary Python packages are in requirements.txt. This pipeline was developed and tested using Python 3.6.8 and on 8GB RAM + 8GB VRAM. The pipeline is composed of 7 steps (0 is optional) and is orchestrated by a snakemake pipeline (`Snakefile`):

0. The model checkpoints are downloaded if they are not available
1. Quality control of the WBS --- each 512*512 tile is quality controlled to filter out tiles with excessive/defficient cellular density and/or poor resolution (output stored as `{output.directory}/_quality_control/{slide.id}.h5`)
2. Segmentation of WBC and RBC --- segmentation coordinates are stored in hdf5 format, with a dataset for each cell (output stored as `{output.directory}/_segmented_wbc/{slide.id}.h5` and `{output.directory}/_segmented_rbc/{slide.id}.h5`)
3. Morphometric characterization of WBC --- morphometric features are stored in hdf5 format, with a dataset for each cell (output stored as `{output.directory}/_aggregates_wbc/{slide.id}.h5`)
4. Morphometric characterization of RBC --- morphometric features are stored in hdf5 format, with a dataset for each cell (output stored as `{output.directory}/_aggregates_rbc/{slide.id}.h5`)
5. Annotation of WBC in geojson format --- these annotations can be loaded into QuPath (output stored as `{output.directory}/_annotations_wbc/{slide.id}.h5`)
6. Annotation of RBC in geojson format --- these annotations can be loaded into QuPath (output stored as `{output.directory}/_annotations_rbc/{slide.id}.h5`)

Steps 3 and 4 are run in parallel, as well as steps 5 and 6. In the examples above, `{slide.id}` is the `basename` of the slide until the first point (if the slide path is `/homes/user/slide_32.0.1.tiff` then `{slide.id}` is `slide_32`). The output directory is specified in the scripts below as `-o`.

## Usage of `run-slide.sh` (to run a single digitalised slide):

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
                sh run-slide.sh -i slide_dir -o output_dir -f tiff -r 1.1
```

## Usage of `run-folder.sh` (to run a folder of digitalised slides with a given format):

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
                sh run-folder.sh -i slide_dir -o output_dir -f tiff -r 1.1
```

