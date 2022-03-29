#!/bin/bash
QC_CKPT=parameters/qc-net
XGB_PATH=parameters/xgboost
UNET_CKPT=parameters/u-net
RF=1.0
MODE=local

TEMP=`getopt --long -o "i:o:q:x:u:r:m:h" "$@"`
eval set -- "$TEMP"
while true
do
    case "$1" in
        -i ) SLIDE_PATH=$2; shift 2 ;;
        -o ) OUTPUT_PATH=$2; shift 2 ;;
        -q ) QC_CKPT=$2; shift 2 ;;
        -x ) XGB_PATH=$2; shift 2 ;;
        -u ) UNET_CKPT=$2; shift 2 ;;
        -r ) RF=$2; shift 2 ;;
        -m ) MODE=$2; shift 2 ;;
        -h )
            echo "Haemorasis - blood cell detection pipeline"
            echo "arguments (required):
                -i : input slide
                -o : output directory"
            echo ""
            echo "arguments (optional):
                -q : path to quality control network checkpoint (if not in -q will be downloaded)
                -u : path to WBC segmentation network checkpoint (if not in -u will be downloaded)
                -x : path to RBC object filtering XGboost model parameters (if not in -u will be downloaded)
                -r : rescale factor (default = 1.0)
                -m : mode (local or cluster (only LSF supported))"
            echo ""
            echo "example:
                sh run-slide.sh -i slide_dir -o output_dir -f tiff -r 1.1"
            echo ""
            exit
        ;;
        * ) break
    esac
done

mkdir -p logs

if [[ $MODE == "cluster" ]]
then
    snakemake \
        --config slide_path=$SLIDE_PATH output_path=$OUTPUT_PATH xgb_path=$XGB_PATH\
        unet_ckpt=$UNET_CKPT qc_ckpt=$QC_CKPT rescale_factor=$RF\
        -s Snakefile \
        --latency-wait 3000 \
        --jobs 3 \
        -p \
        --cluster 'bsub -M {params.mem} -n {params.n_cores} -o logs/{params.log_id}.o -e logs/{params.log_id}.e'
else
   snakemake \
        --config slide_path=$SLIDE_PATH output_path=$OUTPUT_PATH xgb_path=$XGB_PATH\
        unet_ckpt=$UNET_CKPT qc_ckpt=$QC_CKPT rescale_factor=$RF\
        -s Snakefile \
        --latency-wait 3000 \
        --jobs 3 \
        -p
fi
