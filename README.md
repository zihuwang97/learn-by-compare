# Learn-by-Compare (LbC)

## Requirements

See `requirements.txt`

## Datasets

Pre-layout and Post-layout data are preprocessed (coverted to csv files). They can be found under the folder `BFMC-1000-PRE-LAY` and `BFMC-1000-POST-LAY`. `input.csv` and `output.csv` are the parameters and their corresponding two measurements.

## L1 Regression

python main_l1.py --data_path BFMC-1000-PRE-LAY/inputdata.npy --label_path BFMC-1000-PRE-LAY/outputdata.npy --embed_size 80

## LbC

python main_lbc.py --data_path BFMC-1000-PRE-LAY/inputdata.npy --label_path BFMC-1000-PRE-LAY/outputdata.npy --embed_size 80 --temperature 0.1
