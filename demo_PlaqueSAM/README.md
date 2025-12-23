## PlaqueSAM Inference Demo Guide

This document explains how to prepare the input data and run inference with **PlaqueSAM**.
All instructions are simplified and user‑friendly so you can set up the demo quickly.

### Input Data Format
The input data for PlaqueSAM must be organized in the following folder structure:

```python
JPEGImages/
├── Patient_01/
│   ├── 001.jpg
│   ├── 002.jpg
│   ├── 003.jpg
│   ├── 004.jpg
│   ├── 005.jpg
│   └── 006.jpg
├── Patient_02/
│   ├── 001.jpg
│   ├── 002.jpg
│   ├── 003.jpg
│   ├── 004.jpg
│   ├── 005.jpg
│   └── 006.jpg
├── Patient_03/
│   ├── 001.jpg
│   ├── 002.jpg
│   ├── 003.jpg
│   ├── 004.jpg
│   ├── 005.jpg
│   └── 006.jpg
...
```

### Directory Rules
1. JPEGImages/: The root directory containing all patient image data.

2. Patient folders: You may name them freely (e.g., Patient_01, Case_A, ID1234), as long as each folder contains one patient.

3. Image requirements

- Each patient folder must contain exactly 6 images.
- Filenames must follow this fixed pattern:
  
  **001.jpg, 002.jpg, 003.jpg, 004.jpg, 005.jpg, 006.jpg**
 
- Only .jpg files are supported.
- Image resolution can vary.
  
4. Naming restrictions
- Avoid spaces and non‑ASCII characters.
- Use only letters, numbers, and underscores.


### How to Run Inference
Follow the steps below in order.

#### Step 1 — Generate pseudo ground‑truth for inference

```shell
python tools/gene_pseudo_gt_for_inference.py
```

#### Step 2 — Update configuration file

Open `sam2/configs/sam2.1_training/sam2.1_hiera_t+_MOSE_finetune_infer.yaml` and modify:

- Line 15 — `dataset_root_path`

  Set to the absolute path of your JPEGImages/ directory.

- Line 486 — `checkpoint_path`
  
  Path to your PlaqueSAM model checkpoint.

- Line 492 — `experiment_log_dir`
  
  Path to save inference results.

#### Step 3 — Run inference script

```shell
sh run_infer.sh
```

#### Step 4 — Post‑process the model outputs

```shell
python tools/postprocess_for_pred_json_PlaqueSAM.py
```

#### Step 5 — Visualize predicted instance segmentation results

```shell
python tools/ins_seg_json_visualization_for_inference.py
```

#### Done!
After completing all steps, your prediction results and visualizations will be saved to the output directory you specified in Step 2.
