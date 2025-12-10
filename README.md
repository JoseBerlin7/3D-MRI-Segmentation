1. Project Structure
   
        BRATS_3D/
        ├─ data/ (raw NIfTI per subject)
        ├─ notebooks/ (exploration, experimentation, visualization)
        ├─ src/
        │  ├─ data_loaders.py (dataset + transforms)
        │  ├─ model_code.py
        │  ├─ train.py
        │  ├─ model_code.py
        ├─ weights/ (checkpoints + predictions)
        └─ README.md

2. Features
   Full 3D medical segmentation workflow
   SwinUNETR backbone from MONAI   
   Mixed precision training with AMP   
   DiceCELoss + Dice metric evaluation   
   Rich augmentations with MONAI transforms   
   TensorBoard logging   
   Azure ML job submission support   
   Clean modular structure: data, model, training

3. Model
   Model is defined in
   
         src/model_code.py 
      
   It uses:
   
      in_channels=4 (T1, T1ce, T2, FLAIR)   
      out_channels=4   
      Patch embedding size of 48   
      Gradient checkpointing enabled

4. Training
   Training is handled in
   
            src/train.py 
   Training includes:
   
   Mixed precision (AMP)
   AdamW optimizer
   DiceCELoss
   DiceMetric for validation
   TensorBoard logging
   Checkpoint saving to weights/
   Validation every N epochs
   
   Run training locally:
   
         python src/train.py --data_path ./data --epochs 50
   
   
   Arguments:
   
         --data_path: path to BraTS subject folders
         
         --epochs: number of training epochs
   
   Checkpoints land inside the weights/ folder.

5. Running on Azure ML

   To schedule training on Azure ML, use
   
         src/submit_job.py 
   
   This script:
   
   Registers a custom conda environment from environment.yml   
   Submits train.py as a job   
   Uses your configured GPU cluster (NC4as-T4-V3-Cluster)   
   Prints the Azure ML Studio URL for monitoring
   
   Execute:
   
         python src/submit_job.py


   Make sure your Azure credentials and workspace details are configured.

6. Logging & Monitoring

      Training loss and validation Dice are logged with SummaryWriter.
      
      Open TensorBoard:
      
         tensorboard --logdir runs/

7. Requirements

All dependencies are defined in environment.yml.

This includes:
      
      PyTorch
      
      MONAI
      
      NiBabel
      
      Azure ML SDK
      
      TensorBoard

Create your environment:

      conda env create -f environment.yml
      conda activate swinunetr-env


**Note:** Metrics and visualizations can be verified at the notebooks/dev_note.ipynb file
