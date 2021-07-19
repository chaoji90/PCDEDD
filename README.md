# PCD and EDD
Source code for PCD-DARTS and EDD-DARTS

## version: pytorch 1.7.0

## NAS-BENCH-201:  
`
cd AutoDL-Projects_edd/;
bash ./scripts-search/algos/[DARTS-V1_100.sh|DARTS-V1_10.sh] [cifar10|cifar100] 1 -1
cd AutoDL-Projects_pcd/;
bash ./scripts-search/algos/[DARTS-V1_100.sh|DARTS-V1_10.sh] [cifar10|cifar100] 1 -1
`

## DARTS search space:  
`
cd PCDEDD-DARTS/;
python train_search_cyc.py;
python train_search_pl.py
`

## S1-S4 search space:  
`
cd SmoothDARTS/sota/pcdedd;
python train_search_cyc.py --search_space=[s1|s2|s3|s4]
python train_search_pl.py --search_space=[s1|s2|s3|s4] 
`
