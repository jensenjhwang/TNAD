# Anomaly Detection with Tensor Networks

Tensor networks are leveraged as compact representations of large matrices for
anomaly detection.

## Getting Started

Our code is written in Python 3. To install required dependencies, run

```
pip3 install -r requirements.txt
```

## Running Experiments

### Datasets

Image datasets are directly retrieved from the Tensorflow datasets library.
However, for the ODDS datasets, one has store the .mat files in a directory and
subsequently pass the directory path in the `--tab_dir` flag. Ensure that the
file name corresponds to the `--ds_name` flag when running `train.py`.
Currently, the ODDS datasets are stored in `./data/`.

### Running with Flags

Experiments are run via flags specified in the `train.py` file which trains
a model and subsequently evaluates it on the test set. For example, to reproduce
results for MNIST/Fashion-MNIST where only `0` is deemed normal, one
can run

```
python3 train.py --ds_name mnist --alpha 0.4 --spacing 8 --learning_rate 2e-3 --w_decay 0.01 --true_labels 0
```

```
python3 train.py --ds_name fashion_mnist --alpha 0.4 --spacing 8 --learning_rate 2e-3 --w_decay 0.01 --true_labels 0
```

To reproduce results for tabular datasets:

```
python3 train.py --ds_name wine --alpha 0.3 --spacing 1 --p_dim 4 --learning_rate 2e-3 --w_decay 0.01
```

```
python3 train.py --ds_name glass --alpha 0.3 --spacing 2 --p_dim 16 --emb_type fourier --learning_rate 5e-4 --w_decay 5e-4
```

```
python3 train.py --ds_name thyroid --alpha 0.1 --spacing 1 --p_dim 6 --learning_rate 2e-3 --w_decay 0.01
```

```
python3 train.py --ds_name satellite --alpha 0.1 --spacing 2 --p_dim 4 --learning_rate 5e-4 --w_decay 5e-4
```

```
python3 train.py --ds_name cover --alpha 0.1 --spacing 1 --p_dim 8 --learning_rate 5e-4 --w_decay 5e-4
```

## Results

Our results on MNIST/Fashion-MNIST and five ODDS datasets are shown below.

![Alt text](/results/mnist.png?raw=true "MNIST/Fashion-MNIST Results")

![Alt text](/results/odds.png?raw=true "ODDS Results")
