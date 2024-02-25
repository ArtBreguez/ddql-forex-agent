## Running the Code

To train the model, download a training and test csv files from [Yahoo! Finance](https://ca.finance.yahoo.com/quote/%5EGSPC/history?p=%5EGSPC) into `data/`
```
mkdir model
python train EURUSD_train 10 1000
```

Then when training finishes (minimum 200 episodes for results):
```
python evaluate.py EURUSD_test model_ep1000
```
