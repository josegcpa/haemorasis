#!/bin/bash
QC_CKPT=parameters/qc-net
XGB_PATH=parameters/xgboost
UNET_CKPT=parameters/u-net
RF=1.0
MODE=local

TEMP=`getopt --long -o "i:f:o:q:x:u:r:m:h" "$@"`
eval set -- "$TEMP"
while true
do
    case "$1" in
        -i ) DIR=$2; shift 2 ;;
        -f ) FMT=$2; shift 2 ;;
        -o ) OUT=$2; shift 2 ;;
        -q ) QC_CKPT=$2; shift 2 ;;
        -x ) XGB_PATH=$2; shift 2 ;;
        -u ) UNET_CKPT=$2; shift 2 ;;
        -r ) RF=$2; shift 2 ;;
        -m ) MODE=$2; shift 2 ;;
        -h )
            echo "Batch Haemorasis - blood cell detection pipeline"
            echo "arguments (required):
                -i : input directory
                -f : file extension (i.e. ndpi, svs, tiff, etc...)
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
                sh run-folder.sh -i slide_dir -o output_dir -f tiff -r 1.1"
            echo ""
            exit
        ;;
        *)
            break
        ;;
    esac
done

DIR=$(echo $DIR | sed "s/\/$//")
OUT=$(echo $OUT | sed "s/\/$//")

echo "detecting all *$FMT slides in $DIR"
echo "output in $OUT"
echo "  rescale factor is $RF "

for slide_path in $DIR/*$FMT
do
    R=$(basename $slide_path)
    bash run-slide.sh -i $slide_path -o $OUT -q $QC_CKPT -x $XGB_PATH -u $UNET_CKPT -r $RF -m $MODE
done
