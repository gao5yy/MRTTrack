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
Download [LasHeR](https://github.com/BUGPLEASEOUT/LasHeR),[VTUAV](https://zhang-pengyu.github.io/DUT-VTUAV/),[RGBT234](https://sites.google.com/view/ahutracking001/).

## Training
Training from scrath

Download [OSTrack](https://github.com/botaoye/OSTrack). And, training with dataset [Lasot](http://vision.cs.stonybrook.edu/~lasot/), [COCO](https://cocodataset.org/#download), [GOT-10k](http://got-10k.aitestunion.com/) and [TrackingNet](https://github.com/SilvioGiancola/TrackingNet-devkit) for SOT pretrained model. 
（Download [SOT](https://pan.baidu.com/s/1U42J6b3g1htma0OvmXRQCw?pwd=at5b#list/path=%2F) pretrained weights） And put it under ```$PROJECT_ROOT$/pretrained_models```.

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
We refer you to [LasHeR Toolkit](https://github.com/BUGPLEASEOUT/LasHeR) for LasHeR evaluation, and refer you to [MPR_MSR_Evaluation](https://sites.google.com/view/ahutracking001/) for RGBT234 and RGBT210 evaluation.

## Result
|Dataset |Model | Backbone | Pretraining | Precision | NormPrec | Success | FPS |
|---------|---------|----------|-------------|-----------|----------|---------|------|
|LasHeR|MRTrack | ViT-Base | SOT | 70.2 |66.5 |56.5| 32.5|

|Dataset |Model | Backbone | Pretraining | Precision | Success | FPS |
|---------|---------|----------|-------------|-----------|----------|---------|------|
|RGBT210|MRTrack | ViT-Base | SOT | 85.6 |63.1| 32.5|
|RGBT234|MRTrack | ViT-Base | SOT | 87.2 |64.1| 32.5|

|Dataset |Model | Backbone | Pretraining | MPR | MSR | FPS |
|---------|---------|----------|-------------|-----------|----------|---------|------|
|VTUAV-ST|MRTrack | ViT-Base | SOT | 84.3 |72.1| 32.5|

## Acknowledgments
Our project is developed upon [TBSI](https://github.com/RyanHTR/TBSI?tab=readme-ov-file). Thanks for their contributions which help us to quickly implement our ideas.




