## Environment Installation

```bash
conda create -n sts python=3.8
conda activate sts
bash install.sh
```

## Project Paths Setup

Run the following command to set paths for this project
```
python tracking/create_default_local_file.py --workspace_dir . --data_dir ./data --save_dir ./output
```

After running this command, you can also modify paths by editing these two files
```
lib/train/admin/local.py  # paths about training
lib/test/evaluation/local.py  # paths about testing
```

## Data Preparation
Put the tracking datasets in ```./data```. It should look like:
```
${PROJECT_ROOT}
  -- data
      -- lasher
          |-- trainingset
          |-- testingset
          |-- trainingsetList.txt
          |-- testingsetList.txt
          ...
```

## Training
Download [SOT](https://pan.baidu.com/s/1U42J6b3g1htma0OvmXRQCw?pwd=at5b#list/path=%2F) pretrained weights and put them under ```$PROJECT_ROOT$/pretrained_models```.

```bash
python tracking/train.py --script select_track --config vitb_256_select_32x1_1e4_lasher_15ep_sot --save_dir ./output --mode multiple --nproc_per_node 4
```

## Evaluation
Download [checkpoint](https://pan.baidu.com/s/18u2FJu1ZZ7_w-mmSDMEx1A?pwd=eq98) and put it under ```$PROJECT_ROOT$/output```.

```bash
python tracking/test.py select_track vitb_256_select_32x1_1e4_lasher_15ep_sot --dataset_name lasher_test
```

Download [raw result](https://pan.baidu.com/s/1XMDrudiK-kl2cTe76Td2QA?pwd=av9c) and put it under ```$PROJECT_ROOT$/output```.

```bash
python tracking/analysis_results.py
```
We refer you to [LasHeR Toolkit](https://github.com/BUGPLEASEOUT/LasHeR) for LasHeR evaluation, and refer you to [MPR_MSR_Evaluation](https://sites.google.com/view/ahutracking001/) for RGBT234 evaluation.

## Acknowledgments
Our project is developed upon [TBSI](https://github.com/RyanHTR/TBSI?tab=readme-ov-file). Thanks for their contributions which help us to quickly implement our ideas.




