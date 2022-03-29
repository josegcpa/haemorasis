import sys
import os

slide_path = config['slide_path']
output_path = config['output_path']
unet_ckpt = config['unet_ckpt']
qc_ckpt = config['qc_ckpt']
xgb_path = config['xgb_path']
rescale_factor = config['rescale_factor']

tmp_folder = 'tmp'
if "CKPT_URL" in os.environ:
    parameter_url = os.environ["CKPT_URL"]
else:
    raise KeyError("CKPT_URL should be defined as an environment variable")
slide_id = slide_path.split('/')[-1].split('.')[0]

try:
    os.makedirs(output_path)
except:
    pass

dir_dict = {
    "quality_control":"_quality_control",
    "segmented_rbc":"_segmented_rbc",
    "segmented_wbc":"_segmented_wbc",
    "aggregates_rbc":"_aggregates_rbc",
    "aggregates_wbc":"_aggregates_wbc",
    "annotations_wbc":"_annotations_wbc",
    "annotations_rbc":"_annotations_rbc",
    "checkpoints_common":"_checkpoints",
    "checkpoints":"_checkpoints"}

for k in dir_dict:
    try:
        folder = dir_dict[k]
        os.makedirs(os.path.join(output_path,folder))
    except:
        pass

localrules: all, download_checkpoints, quality_control, segment_wbc_rbc

rule all:
    input:
        os.path.join(output_path,dir_dict["aggregates_rbc"],slide_id+".h5"),
        os.path.join(output_path,dir_dict["aggregates_wbc"],slide_id+".h5"),
        os.path.join(output_path,dir_dict["annotations_rbc"],slide_id+".geojson"),
        os.path.join(output_path,dir_dict["annotations_wbc"],slide_id+".geojson"),
        os.path.join(output_path,dir_dict["checkpoints"],slide_id+"_agg_wbc"),
        os.path.join(output_path,dir_dict["checkpoints"],slide_id+"_agg_rbc"),
        os.path.join(output_path,dir_dict["checkpoints"],slide_id+"_ann_rbc"),
        os.path.join(output_path,dir_dict["checkpoints"],slide_id+"_ann_wbc")

rule download_checkpoints:
    output:
        unet_ckpt=directory(unet_ckpt),
        qc_ckpt=directory(qc_ckpt),
        xgb_path=directory(xgb_path),
        xgb_path_model=os.path.join(xgb_path,"rbc_xgb_model"),
        xgb_scaler_model=os.path.join(xgb_path,"rbc_scaler_params"),
    shell:
        """
        wget -O parameters.zip {parameter_url}
        mkdir -p {tmp_folder}
        unzip -o parameters.zip -d {tmp_folder}

        mkdir -p {output.unet_ckpt} {output.qc_ckpt} {output.xgb_path}
        unzip -o {tmp_folder}/u-net-parameters.zip -d {output.unet_ckpt}
        unzip -o {tmp_folder}/quality-net-parameters.zip -d {output.qc_ckpt}
        unzip -o {tmp_folder}/xgboost-parameters.zip -d {output.xgb_path}
        rm parameters.zip
        rm {tmp_folder}/*zip
        """

rule quality_control:
    input:
        slide_path=slide_path,
        qc_ckpt=qc_ckpt
    output:
        qc_out=os.path.join(output_path,dir_dict["quality_control"],slide_id),
        checkpoint_qc=os.path.join(output_path,dir_dict["checkpoints_common"],slide_id+"_qc")
    message:
        "Running slide through QC network."
    params:
        qc_ckpt=qc_ckpt
    shell:
        """
        python3 scripts/python/quality_control.py \
         --slide_path {input.slide_path}\
         --input_height 512\
         --input_width 512\
         --checkpoint_path {params.qc_ckpt}\
         --batch_size 32 > {output.qc_out} && touch {output.checkpoint_qc}
        """

rule segment_wbc_rbc:
    input:
        slide_path=slide_path,
        qc_out=os.path.join(output_path,dir_dict["quality_control"],slide_id),
        checkpoint_qc=os.path.join(output_path,dir_dict["checkpoints_common"],slide_id+"_qc")
    output:
        seg_rbc=os.path.join(output_path,dir_dict["segmented_rbc"],slide_id+".h5"),
        seg_wbc=os.path.join(output_path,dir_dict["segmented_wbc"],slide_id+".h5"),
        checkpoint_seg=os.path.join(output_path,dir_dict["checkpoints"],slide_id+"_seg")
    message:
        "Segmenting WBC and RBC."
    params:
        log_id="WBC_RBC_SEGMENTATION_{}".format(slide_id),
        n_cores=8,
        mem=16000,
        unet_ckpt_path=os.path.join(unet_ckpt,"u-net"),
        depth_mult=0.5,
        rescale_factor=rescale_factor
    shell:
        """
        python3 scripts/python/segment_slide_wbc_rbc.py \
            --csv_path {input.qc_out} \
            --slide_path {input.slide_path} \
            --unet_checkpoint_path {params.unet_ckpt_path} \
            --wbc_output_path {output.seg_wbc} \
            --rbc_output_path {output.seg_rbc} \
            --depth_mult {params.depth_mult} \
            --rescale_factor {params.rescale_factor} && touch {output.checkpoint_seg}
        """

rule characterise_aggregate_rbc:
    input:
        slide_path=slide_path,
        seg_rbc=os.path.join(output_path,dir_dict["segmented_rbc"],slide_id+".h5"),
        checkpoint_seg=os.path.join(output_path,dir_dict["checkpoints"],slide_id+"_seg"),
        xgb_model_path=os.path.join(xgb_path,"rbc_xgb_model"),
        xgb_scaler_path=os.path.join(xgb_path,"rbc_scaler_params")
    output:
        agg_rbc=os.path.join(output_path,dir_dict["aggregates_rbc"],slide_id+".h5"),
        checkpoint_agg_rbc=os.path.join(output_path,dir_dict["checkpoints"],slide_id+"_agg_rbc")
    message:
        "Characterising and aggregating RBC."
    params:
        log_id="RBC_CHARACTERISATION_AGGREGATION_{}".format(slide_id),
        n_cores=8,
        mem=8000,
        rescale_factor=rescale_factor
    shell:
        """
        python3 scripts/python/characterise_cells.py \
            --slide_path {input.slide_path} \
            --segmented_cells_path {input.seg_rbc} \
            --cell_type rbc \
            --n_processes {params.n_cores} \
            --rescale_factor {params.rescale_factor} \
            --standardizer_params {input.xgb_scaler_path} \
            --xgboost_model {input.xgb_model_path} \
            --output_path {output.agg_rbc} && touch {output.checkpoint_agg_rbc}
        """

rule characterise_aggregate_wbc:
    input:
        slide_path=slide_path,
        seg_wbc=os.path.join(output_path,dir_dict["segmented_wbc"],slide_id+".h5"),
        checkpoint_seg=os.path.join(output_path,dir_dict["checkpoints"],slide_id+"_seg")
    output:
        agg_wbc=os.path.join(output_path,dir_dict["aggregates_wbc"],slide_id+".h5"),
        checkpoint_agg_wbc=os.path.join(output_path,dir_dict["checkpoints"],slide_id+"_agg_wbc")
    message:
        "Characterising and aggregating WBC."
    params:
        log_id="WBC_CHARACTERISATION_AGGREGATION_{}".format(slide_id),
        n_cores=8,
        mem=8000,
        rescale_factor=rescale_factor
    shell:
        """
        python3 scripts/python/characterise_cells.py \
            --slide_path {input.slide_path} \
            --segmented_cells_path {input.seg_wbc} \
            --cell_type wbc \
            --n_processes {params.n_cores} \
            --rescale_factor {params.rescale_factor} \
            --output_path {output.agg_wbc} && touch {output.checkpoint_agg_wbc}
        """

rule annotate_wbc:
    input:
        seg_wbc=os.path.join(output_path,dir_dict["segmented_wbc"],slide_id+".h5"),
        checkpoint_seg=os.path.join(output_path,dir_dict["checkpoints"],slide_id+"_seg"),
        agg_wbc=os.path.join(output_path,dir_dict["aggregates_wbc"],slide_id+".h5"),
        checkpoint_agg_wbc=os.path.join(output_path,dir_dict["checkpoints"],slide_id+"_agg_wbc")
    output:
        ann_wbc=os.path.join(output_path,dir_dict["annotations_wbc"],slide_id+".geojson"),
        checkpoint_ann_wbc=os.path.join(output_path,dir_dict["checkpoints"],slide_id+"_ann_wbc")
    message:
        "Annotating WBC"
    params:
        log_id="WBC_ANNOTATION_{}".format(slide_id),
        n_cores=2,
        mem=8000,
        colour="93 63 211",
        name="wbc"
    shell:
        """
        python3 scripts/python/cells-to-annotations.py \
            --segmented_cells_path {input.seg_wbc} \
            --characterized_cells_path {input.agg_wbc} \
            --name {params.name} \
            --colour {params.colour} \
            --output_geojson {output.ann_wbc} && touch {output.checkpoint_ann_wbc}
        """

rule annotate_rbc:
    input:
        seg_rbc=os.path.join(output_path,dir_dict["segmented_rbc"],slide_id+".h5"),
        checkpoint_seg=os.path.join(output_path,dir_dict["checkpoints"],slide_id+"_seg"),
        agg_rbc=os.path.join(output_path,dir_dict["aggregates_rbc"],slide_id+".h5"),
        checkpoint_agg_rbc=os.path.join(output_path,dir_dict["checkpoints"],slide_id+"_agg_rbc")
    output:
        ann_rbc=os.path.join(output_path,dir_dict["annotations_rbc"],slide_id+".geojson"),
        checkpoint_ann_rbc=os.path.join(output_path,dir_dict["checkpoints"],slide_id+"_ann_rbc")
    message:
        "Annotating RBC"
    params:
        log_id="RBC_ANNOTATION_{}".format(slide_id),
        n_cores=2,
        mem=8000,
        colour="200 0 0",
        name="rbc"
    shell:
        """
        python3 scripts/python/cells-to-annotations.py \
            --segmented_cells_path {input.seg_rbc} \
            --characterized_cells_path {input.agg_rbc} \
            --name {params.name} \
            --colour {params.colour} \
            --output_geojson {output.ann_rbc} && touch {output.checkpoint_ann_rbc}
        """

