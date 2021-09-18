# Graph Convolutional Network

Predicting solubility of a molecule in water using graph convolutional networks

Installtion

```bash
pip install -r requirements.txt
```
```bash
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.6.0+cu102.html
```
The above command has to be modified in case you have different version of pytorch and cuda installed

Train GCN

```bash
python3 train.py
```

Run inference

```bash
python3 predict.py
```
The above command runs prediction on one graph (a molecule) and returns its solubility in water

