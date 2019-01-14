# Deep Learning for the classification of Sentinel-2 image time series

Training temporal Convolution Neural Netoworks (TempCNNs), Recurrent Neural Networks (RNNs) and Random Forests (RFs) on satelitte image time series.
This code is supporting a paper submitted to IEEE International Geoscience and Remote Sensing Symposium (IGARSS) 2019:
```
@inproceedings{Pelletier2018Deep,
    Title = {Deep Learning for the classification of Sentinel-2 image time series},
    Author = {Pelletier, Charlotte and Webb, Geoffrey I and Petitjean, Francois},
    Booktitle = {Submitted to IEEE International Geoscience and Remote Sensing Symposium (IGARSS) 2019},
    note = {In revision}
}
```

## Examples

### Running the models

- training TempCNNs: `python train_classifier.py --classifier TempCNN --train train_dataset.csv --test test_dataset.csv`
- training bidirectional GRU-RNNs: `python train_classifier.py --classifier GRU-RNNbi --train train_dataset.csv --test test_dataset.csv`
- training GRU-RNNs: `python train_classifier.py --classifier GRU-RNN --train train_dataset.csv --test test_dataset.csv`
- training RFs: `python train_classifier.py --classifier RF --train train_dataset.csv --test test_dataset.csv`

It will output a result file including the OA computed on test data, the confusion matrix, the training history for deep learning models, and the learned model.

Each model will be trained on `train_dataset.csv` file and test on `test_dataset.csv` file.  
Please note that both `train_dataset.csv` and `test_dataset.csv` files are a subsample of the data used in the paper: original data cannot be distributed.

Thoses files have an header, and contain one observation per row having the following format:
`[class,objectID,date1.B2,date1.B3,date1.B4,date1.B5,date1.B6,date1.B7,date1.B8,date1.B8A,date1.B11,date1.B12,...,date73.B12]`

## Contributors
 - [Dr. Charlotte Pelletier](https://sites.google.com/site/charpelletier)
 - [Professor Geoffrey I. Webb](http://i.giwebb.com/)
 - [Dr. Francois Petitjean](http://www.francois-petitjean.com/Research/)
