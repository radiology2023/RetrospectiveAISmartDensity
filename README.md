# AISmartDensity

The collection of Python scripts in this repository is designed for AISmartDensity Project.

Folder `cancer_risk` contains scripts on data processing, and model training of two componenet models: cancer signs and inherent risk.

Folder `masking` contains scripts on model training of the third component model - masking model. 

## Data preprocessing
`cancer_risk/data.py`
- Reads DICOM mammogram images
- Implements image processing including flipping and intensity rescaling for the images based on their DICOM characteristics
- Applies a custom cropping method based on distance transform and pads or crops images to a specified size

To run the script, 
```
python data.py
```

## Cancer signs and inherent risk model training
`cancer_risk/train.py`
- Trains models for cancer detection or risk assessment using EfficientNetB3 architecture
- Includes optional resnet block addition for cancer model training
- Utilizes TensorFlow's tf.distribute.MirroredStrategy() for distributed training across multiple GPUs
- Saves model checkpoints and logs training metrics for tensorboard visualization

To run the script, 
```
python train.py \
  --data_folder "path/to/data_folder" \
  --model_dir "path/to/model_dir" \
  --model_type "cancer" \  # or "risk"
  --num_epoch 50 \
  --img_height 1024 \
  --img_width 832 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --checkpoint_option "noisy-student"
```

## Masking potential model training
`masking/main.py` 
- Executes training ResNet-34 models with configurable parameters
- Utilizes PyTorch for model operations
- Integrates with Weights & Biases (wandb) for experiment tracking

To start model training, 
```
python main.py --train \
  --model_name "my_model" \
  --loss_type "cross_entropy" \
  --train_folder "path/to/train_data" \
  --train_csv "path/to/train_labels.csv" \
  --n_epochs 50 \
  --gpu_id 0
```

