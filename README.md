# DISC

This repository contains source codes and datasets for the paper:

- DISC: Disentangling Spurious Correlations for Multi-Behavior Recommendation

## Usage
### Train & Test

- Training DISC on IJCAI15:
```shell
python main.py --dataset=IJCAI_15
```

- Training DISC on Tmall:
```shell
python main.py --dataset=Tmall
```

- Training DISC on Retail:
```shell
python main.py --dataset=retailrocket
```

- Testing DISC using a saved model file:
```shell
ipython evaluation.ipynb
```
