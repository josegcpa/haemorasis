source ./config
output_path_base=$RESULTS_DIR/blur-maps

output_path=$output_path_base/MLL
mkdir -p $output_path
for slide in $MLL_DIR/*$MLL_FMT
do
    o=$output_path/$(basename $slide | cut -d '.' -f 1).csv
    bsub \
        -n 1 \
        -M 1000 \
        -o /dev/null -e /dev/null \
        "python3 scripts/python/get_slide_blur.py --slide_path $slide > $o"
done

output_path=$output_path_base/ADDEN1
mkdir -p $output_path
for slide in $ADDEN1_DIR/*$ADDEN1_FMT
do
    o=$output_path/$(basename $slide | cut -d '.' -f 1).csv
    bsub \
        -n 1 \
        -M 1000 \
        -o /dev/null -e /dev/null \
        "python3 scripts/python/get_slide_blur.py --slide_path $slide > $o"
done

output_path=$output_path_base/ADDEN2
mkdir -p $output_path
for slide in $ADDEN2_DIR/*$ADDEN2_FMT
do
    o=$output_path/$(basename $slide | cut -d '.' -f 1).csv
    bsub \
        -n 1 \
        -M 1000 \
        -o /dev/null -e /dev/null \
        "python3 scripts/python/get_slide_blur.py --slide_path $slide > $o"
done
