# GraphLSTA
This is a PyTorch implementation of GraphLSTA

`model.py`, `block.py` and `layer.py` contain the key code of the GraphLSTA algorithm

## Requirements
* PyTorch = 1.7.0
* Numpy = 1.18.5
* Sklearn = 0.23.1

## Run the demo

The main function is in the `main.py`

In `main.py`, you can edit the variable `dataName` and `anomaly_per` to choose the running datasets

## Dataset deatial 

`dataConfig.json` record the hyperparameter setting of all the datasets

The datasets of this demo include:
* `UCI` = UCI Message
* `Digg` = Digg
* `alp` = Bitcoin-alpha
* `otc` = Bitcoin-OTC
* `email-d` = Email-DNC
* `email-u`= Email-Eu

Each dataset has three version, which refers to 10%, 5% and 1% anomaly injection percent
