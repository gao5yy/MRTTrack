## Environment Installation

```bash
conda create -n tbsi python=3.8
conda activate tbsi
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
