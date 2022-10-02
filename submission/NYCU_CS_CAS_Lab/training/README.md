## Prepare environment
- install required python modules
    - python version == 3.8
    - cuda version == 10.1
    ```
    pip install -r requirements.txt
    ```
## Prepare dataset
Please put dataset in `./training/dataset/` .

```
# Directory tree
./training
│  dataloader.py
│  model.py
│  README.md
│  requirements.txt
│  training.py
│
├─dataset
│      Please_put_dataset_here
│      S01-AFb-1.txt
│      S01-AFb-2.txt
│      ...
│
├─data_indices
│      test_indice.csv
│      train_indice.csv
│
└─trained_models
```

## Train
Training script entrance is `./training/training.py`.

```
# PWD == './training/'
python training.py
```
The trained model will be in `./training/trained_models/` 