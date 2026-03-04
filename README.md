# 0. DATASET OVERVIEW
From [UrbanSound8K Dataset](https://www.kaggle.com/code/amenmohamed/environmental-sound-classification), **8,732 labeled sound excerpts, each with a duration of 4 seconds or less, categorized into 10 urban sound classes** 


아래는 `"Class: air_conditioner (ID: 0)"` 에 대한 예시
<img width="1175" height="290" alt="Image" src="https://github.com/user-attachments/assets/e1822da8-058d-45b9-a0ea-69b89359f61a" />

### 📊 Class Distribution (UrbanSound8K Subset)

| Class              | Count |
|--------------------|-------|
| dog_bark           | 1000  |
| children_playing   | 1000  |
| air_conditioner    | 1000  |
| street_music       | 1000  |
| engine_idling      | 1000  |
| jackhammer         | 1000  |
| drilling           | 1000  |
| siren              | 929   |
| car_horn           | 429   |
| gun_shot           | 374   |


### 📄 Dataset Metadata 

| slice_file_name      | fsID   | start | end       | salience | fold | classID | class             |
|----------------------|--------|-------|-----------|----------|------|---------|-------------------|
| 100032-3-0-0.wav     | 100032 | 0.0   | 0.317551  | 1        | 5    | 3       | dog_bark          |
| 100263-2-0-117.wav   | 100263 | 58.5  | 62.500000 | 1        | 5    | 2       | children_playing  |
| 100263-2-0-121.wav   | 100263 | 60.5  | 64.500000 | 1        | 5    | 2       | children_playing  |
| 100263-2-0-126.wav   | 100263 | 63.0  | 67.000000 | 1        | 5    | 2       | children_playing  |
| 100263-2-0-137.wav   | 100263 | 68.5  | 72.500000 | 1        | 5    | 2       | children_playing  |



# 1. Environment Setup ⚙️

For venv users
```
python3.10 -m venv .venv 
source .venv/bin/activate 
pip3 install -r requirements.txt 
```

For conda users 
```
conda create -n venv 
conda activate venv
pip3 install -r requirements.txt
```

For windows/git bash users
```
python3 -m venv .venv
source .venv/Scripts/activate
pip3 install -r requirements.txt
```


# 2. Download and preprocess the dataset 📀
```
python -m src.data.process
```

# 3. Training 🔥
```
#train for scratch
python3 -m src.scripts.train_scratch

#train for full fine-tuned model
python3 -m src.scripts.train_finetune

#train for LoRA
python3 -m src.scripts.train_lora

#train for Adapter
python3 -m src.scripts.train_adapter
```

# 4. Compare the results 📈
## 4-1. Run on local / terminal
```
python3 -m src.scripts.compare
```

## 4-2. Run on Google Colab
```python
%cd /content/demo-repository
import sys
sys.path.append("/content/demo-repository")

from src.scripts.compare import main as compare_main
compare_main()
```

# 5. References 📕
[CNN](https://arxiv.org/pdf/1312.4400v3), [LoRA](https://arxiv.org/pdf/2106.09685), [Adapter](https://openaccess.thecvf.com/content/CVPR2024W/PV/papers/Chen_Conv-Adapter_Exploring_Parameter_Efficient_Transfer_Learning_for_ConvNets_CVPRW_2024_paper.pdf)
