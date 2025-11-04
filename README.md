1. Project Structure
   
        BRATS_3D/
        ├─ data/ (raw NIfTI per subject)
        ├─ notebooks/ (exploration + visualization)
        ├─ model/
        │  ├─ data.py (dataset + transforms)
        │  ├─ model.py
        │  └─ utils.py (metrics, visualization)
        ├─ train.py
        ├─ results/ (checkpoints + predictions)
        └─ README.md
