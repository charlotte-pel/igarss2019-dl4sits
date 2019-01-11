# Deep Learning for the classification of Sentinel-2 image time series

Training temporal Convolution Neural Netoworks (TempCNNs), Recurrent Neural Networks (RNNs) and Random Forests (RFs) on satelitte image time series.
This code is supporting a paper submitted at IEEE International Geoscience and Remote Sensing Symposium (IGARSS) 2019


More information about our research at https://sites.google.com/site/charpelletier, http://www.francois-petitjean.com/Research/, and http://i.giwebb.com/

## Examples

### Running the models

- training TempCNNs: `python train_tempcnn.py`
- training RNNs: `python train_rnn.py`
- training RFs: `python train_rf.py`

It will output a result file including the OA on test data, the training history for deep learning models, and the learned model.

Each model will be trained on `train_dataset.csv` file and test on `test_dataset.csv` file.  
Please note that both `train_dataset.csv` and `test_dataset.csv` files are a subsample of the data used in the paper: original data cannot be distributed.

Thoses files have an header, and contain one observation per row having the following format:
`[class,date1.B2,date1.B3,date1.B4,date1.B5,date1.B6,date1.B7,date1.B8,date1.B8A,date1.B11,date1.B12,...,date73.B12]`
