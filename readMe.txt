CIFAR-10 Noisy labels challenge:

This project contains the source code for training and evaluating NN on small subset of CIFAR-10 as trainset with noisy labels and the original testset.

For creating the conda environment you can use the environment.yml included. Modify the environment path in the file and run: conda env create -f environment.yml

Project files (for further documentation check the remarks in the code):
train.py - contains the entry point to the project. It loads the configuration, builds the models and runs the train and evaluation process. It is also responsible for results logging.
loss.py - implementation of the symmetric cross entropy loss which is the default lost of the models. Comparing to cross entropy it deals better with overfitting to noisy labels of
"soft" classes and under fitting of "hard" classes. For further details: https://arxiv.org/abs/1908.06112
noisy_dataset.py - the dataset implementation as described in the challenge and also the dataset generator and loader.
utils - package with some utility functions for the train/eval process.
models - package with different model implementations.
logs - contains the run logs. default log file is SCE_stable_v1.log

Default running: python -m train
This way the default configuration - which gave the optimal results - will be loaded to the train / eval procees.

Optional configurations (for string params, please write in lower case):

lr: default=0.01 (start learning rate)
l2_reg: default=5e-4 (L2 regularization value)
grad_bound: default=5.0 (max grad norm for clipping)
train_log_every: default=100 (number of steps between logging)
batch_size: default=128 (self explained...)
data_path': default='../../datasets' (path for downloading CIFAR-10 dataset)
data_nums_workers: default=8 (number of parallel data loaders)
epoch: default=300 (number of epochs)
loss: default='SCE' (either 'SCE' for symmetric cross enthropy or 'CE' for cross enthropy)
alpha: default=1.0 (alpha scale for SCE)
beta: default=1.0 (beta scale for SCE)
version: default='SCE_stable_v1' (Version of run. Will be the log file name)
model: default='scenet' (Name of the model architecture to be used. You can use many variant of classification ConvNets including different variants of
ResNet / VGG / ResNext / MobileNet / GoogleNet / DenseNet etc. Just remember to write the net name in lower case :-). 
'scenet' is the implementation from the paper official github and apparently it gave the best results. For further details look in the models module.)
optim: default='sgd' (Name of the model optimizer to be used: adam / sgd / adagrad / rmsprop)
seed: default=123 (tensors seed)

Have fun :) 

