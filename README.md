# 2022 Intelligent Manufacturing Big Data Analysis Competition Final Round - Project A
Anomaly number of UV oven lamps prediction.


## Train1 Round
- Setup </br>
`mkdir processed`, `mkdir output`

- Copy train1 data from readonly directory `/TOPIC/projectA/` to `./data/raw/train1/` </br>
`python -m check_tr1 --data-dir /TOPIC/projectA/` (run for competition environment only)

- Generate fake data (train2 and test) using train1 data </br>
`python -m gen_fake_data --data-dir ./data/raw/train1/`

- Process train1 data </br>
`python -m dp --data-type tr1`

- Train models using processed train1 data with 10 seeds </br>
`python -m train_eval --proc-file train_1.csv --n-seeds 10`

## Train2 Round
- Check train2 data format </br>
`python -m check_tr2 --data-dir ./data/fake_projectA/`

- Process train1&train2 data </br>
`python -m dp --data-type tr2`

- Train models using processed train1&train2 data with 10 seeds </br>
`python -m train_eval --proc-file train_1_2.csv --n-seeds 10`

## Inference
- Process test data </br>
`python -m dp --data-type ts --ts-source tr_12`
- Inference </br>
`python -m infer --model-dir ./output/fold --save-dir ./output/ --save-fname 111011_projectA_ans.csv`


### Project Structure
```
├── check_tr1.py
├── check_tr2.py
├── config
│   └── model
│       └── lgbm_template.yaml
├── data
│   ├── fake_projectA
│   │   ├── test
│   │   │   └── accumulation_hour3.csv
│   │   └── train2
│   │       ├── accumulation_hour2.csv
│   │       └── anomaly_train2.csv
│   ├── processed
│   │   ├── feat_cols.csv
│   │   ├── oid2idx.pkl
│   │   ├── test.csv
│   │   ├── train_1_2.csv
│   │   └── train_1.csv
│   └── raw
│       ├── test
│       │   ├── accumulation_hour3.csv
│       │   ├── cooler.csv
│       │   └── projectA_template.csv
│       ├── train1
│       │   ├── accumulation_hour1.csv
│       │   ├── anomaly_train1.csv
│       │   ├── A_raw_data.zip
│       │   ├── cooler.csv
│       │   └── power.csv
│       └── train2
│           ├── accumulation_hour2.csv
│           └── anomaly_train2.csv  
├── dp.py
├── gen_fake_data.py
├── infer.py
├── modeling
│   └── build.py
├── notebooks
│   └── simple_eda.ipynb
├── output
│   ├── 111011_projectA_ans.csv   
│   └── fold
|       ├── seed0_fold0.pkl
│       ├── seed0_fold1.pkl
│       ├── seed0_fold2.pkl
│       ├── seed0_fold3.pkl
│       ├── seed0_fold4.pkl
│       ├── seed0_fold5.pkl
│       ├── seed1_fold0.pkl
│       ├── seed1_fold1.pkl
│       ├── seed1_fold2.pkl
│       ├── seed1_fold3.pkl
│       ├── seed1_fold4.pkl
|       ...
├── pyproject.toml
├── README.md
├── setup.cfg
├── tools
│   └── train_eval.py
├── train_eval.py
└── utils
    └── utils.py
```