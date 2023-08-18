## Introduction

This repository is a final project thesis for the Trinity College Dublin Master. 
The project is developed based on the [SpeechDrivesTemplates](https://github.com/ShenhanQian/SpeechDrivesTemplates) project, 
and multi-level processing of audio features is carried out on the basis of the original author. 
Semantic information and Rhythm information are extracted and fused with the original speech features using a self-attention mechanism.

## Installation

To generate videos, you need `ffmpeg` in your system.

```shell
sudo apt install ffmpeg
```

Install Python packages

```shell
pip install -r requirements.txt
```

## Training SDT-BP

**Training** from scratch

``` bash
python main.py --config_file configs/voice2pose_sdt_bp.yaml \
    --tag oliver \
    DATASET.SPEAKER oliver \
    SYS.NUM_WORKERS 32
```

- `--tag` set the name of the experiment which wil be displayed in the outputfile.
- You can overwrite any parameter defined in `configs/default.py` by simply
adding it at the end of the command. The example above set `SYS.NUM_WORKERS` to 32 temporarily.

Resume **training** from an interrupted experiment

``` bash
python main.py --config_file configs/voice2pose_sdt_bp.yaml \
    --resume_from <checkpoint-to-continue-from> \
    DATASET.SPEAKER oliver
```

- With `--resume_from`, the program will load the `state_dict` from the checkpoint for both the model and the optimizer, and write results to the original directory that the checkpoint lies in.

**Training** from a pretrained model

``` bash
python main.py --config_file configs/voice2pose_sdt_bp.yaml \
    --pretrain_from <checkpoint-to-pretrain-from> \
    --tag oliver \
    DATASET.SPEAKER oliver
```

- With `--pretrain_from`, the program will only load the `state_dict` for the model, and write results to a new base directory.

## Evaluation

To **evaluate** a model, use `--test_only` and `--checkpoint` as follows

``` bash
python main.py --config_file configs/voice2pose_sdt_bp.yaml \
    --tag oliver \
    --test_only \
    --checkpoint <path-to-checkpoint> \
    DATASET.SPEAKER oliver
```

## Demo

To **evaluate** a model on an audio file, use `--demo_input` and `--checkpoint` as follows

```bash
python main.py --config_file configs/voice2pose_sdt_bp.yaml \
    --tag oliver \
    --demo_input demo_audio.wav \
    --checkpoint <path-to-checkpoint> \
    DATASET.SPEAKER oliver
```
