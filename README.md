# Introduction

Needles in Haystacks is the accompanying code repository for the paper titled, "Needles in Haystacks: On Classifying Tiny Objects in Large Images" by  Nick Pawlowski, Suvrat Bhooshan, Nicolas Ballas, Francesco Ciompi, Ben Glocker, Michal Drozdzal.

It is provided to replicate the datasets and results of the experiments described in it.  
- **Generating needle MNIST (nMNIST)**
  
- **Generating needle CAMELYON (nCAMELYON)**
  
- **Reproducing the train results on both datasets.**

- **Getting Saliency Map Results**


## Getting Started

### Setup Enviroment


* [PyTorch](http://pytorch.org/) version >= 1.0.0
* Python version >= 3.5
* ``` pip install scikit-image seaborn tensorboardX tensorboard protobuf pandas numpy tqdm imageio torchvision jupyter pytorch scipy```


### Generating Data:

#### nMNIST
To generate data run `data/cluttered_mnist/generate_data.py`. Here `<path/to/raw/mnist/directory>` will be the directory to find the binary MNIST files. They will be downloaded if not yet present.

```
python data/cluttered_mnist/generate_data.py --digit-clutter --distortion-num 2 --patch-size 64 --type detection --canvas 64 --out /checkpoint/jromoff/needle/data/mnist/64_28 --mnist <path/to/raw/mnist/directory>
python data/cluttered_mnist/generate_data.py --digit-clutter --distortion-num 6 --patch-size 64 --type detection --canvas 128 --out /checkpoint/jromoff/needle/data/mnist/128_28 --mnist <path/to/raw/mnist/directory>
python data/cluttered_mnist/generate_data.py --digit-clutter --distortion-num 25 --patch-size 64 --type detection --canvas 256 --out /checkpoint/jromoff/needle/data/mnist/256_28 --mnist <path/to/raw/mnist/directory>
python data/cluttered_mnist/generate_data.py --digit-clutter --distortion-num 100 --patch-size 64 --type detection --canvas 512 --out /checkpoint/jromoff/needle/data/mnist/512_28 --mnist <path/to/raw/mnist/directory>
python data/cluttered_mnist/generate_data.py --digit-clutter --distortion-num 400 --patch-size 64 --type detection --canvas 1024 --out /checkpoint/jromoff/needle/data/mnist/1024_28 --mnist <path/to/raw/mnist/directory>
python data/cluttered_mnist/generate_data.py --digit-clutter --distortion-num 1600 --patch-size 64 --type detection --canvas 2048 --out /checkpoint/jromoff/needle/data/mnist/2048_28 --mnist <path/to/raw/mnist/directory>
python data/cluttered_mnist/generate_data.py --digit-clutter --distortion-num 6400 --patch-size 64 --type detection --canvas 4096 --out /checkpoint/jromoff/needle/data/mnist/4096_28 --mnist <path/to/raw/mnist/directory>
python data/cluttered_mnist/generate_data.py --digit-clutter --distortion-num 14400 --patch-size 64 --type detection --canvas 6144 --out /checkpoint/jromoff/needle/data/mnist/6144_28 --mnist <path/to/raw/mnist/directory>
```

#### nCAMELYON
You have to generate positive and negative crops from the original CAMELYON17 dataset. (https://camelyon17.grand-challenge.org/)
Download the original dataset, and run extract_basic_pos_crops.py and extract_basic_neg_crops.py after setting the labels_path, base_path, and data args in these files.

An Example:

```
python extract_basic_pos_crops.py --size 512 --low_roir 0.1 --high_roir 0.5
python extract_basic_neg_crops.py --size 256
```


### Model Training

train.py provides the common training pipeline for both datasets.  

train_mnist.py and train_camelyon.py provide the necessary wrappers for training on their respective datasets.

Example Training Commands:

```
python train_camelyon.py --pos_path /path/to/positive_crops --neg_path /path/to/negative_crops --logdir /path/to/logs -b 8 -e 50 --delayed_step 4
python train_mnist.py --data_path /path/to/nMNIST_dataset --logdir /path/to/logs -b 32 -e 200 --delayed_step 4
```  


### Saliency Map Results

Run predict_saliency.py to save saliency maps.
Example Usage:

```
python predict_saliency.py --logdir /path/to/model_dir --datatype mnist --data_path /path/to/nMNIST_dataset
python predict_saliency.py --logdir /path/to/model_dir --datatype camelyon --pos_path /path/to/positive_crops --neg_path /path/to/negative_crops  

```

# License

Needles in Haystacks is licensed under Creative Commons-Non Commercial 4.0. See the LICENSE file for details.

# Citation

Please cite it as follows:

```
@article{needles2019,
  author    = {Pawlowski, Nick and Bhooshan, Suvrat and Ballas, Nicolas and Ciompi, Francesco and Glocker, Ben and Drozdzal, Michal},
  title     = "{Needles in Haystacks: On Classifying Tiny Objects in Large Images}",
  journal   = {arXiv preprint},
  year      = {2019},
}
```
