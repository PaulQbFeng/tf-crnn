# Convolutional Recurrent Neural Network in Tensorflow (tf.crnn)
CRNN model in Tensorflow using Estimators

Implementation of the Convolutional Recurrent Neural Network (CRNN) for image-based sequence recognition tasks, such as scene text recognition and OCR.
Original paper http://arxiv.org/abs/1507.05717 and code https://github.com/bgshih/crnn

This version uses the `tf.estimator.Estimator` to build the model.

## Installation

This package requires `python3` and TensorFlow. While CPU version could work, the high computational demands mean that the GPU version is necessary for training.

Currently it has been tested only under `python v3.6.6` since this is the only `py3.6` version that official TensorFlow distribution supports. More compatibility to come.

You may already have a working version of tensorflow, which is why we do not try to automatically install it for you. If you do not, please install it before installing this package:


```bash
$ pip install tensorflow-gpu
```

Then, you can install this package. From the root directory, run:

```bash
$ pip install -r requirements.txt
```

This will symlink the package in the python installation directory, so that if you make modifications, they will instantly be available. Note that installation is not necessary for a simple run, but then all commands should be run from the root directory (because Python automatically discovers packages in the current directory).

## Training

To train, you need to provide a parameters file. An example one is `model_params.json`. You should modify at least the following paths in there:

*  `"output_model_dir": "/path/to/where/model/will/save/weights/"` <-- if same as a previous training, the model will first load weights from here;
* `"tfrecords_train": "/path/to/training/directory/trainig_files_batch*.tfrecords"` <-- a directory where you have the training dataset in `tfrecords` format. It can be a single file, or a `glob` expression
*  `"tfrecords_eval": "/path/to/eval/directory/eval_files_batch*.tfrecords"` <-- same as above, but for validation dataset

Documentation for other parameters: TODO. Hopefully the names are clear for now :)

Then, you can start training with: `python -m tf_crnn.train <path_to_model_params.json>`

You can quickly modify the output directory, the GPU being used or the number of epochs by providing optional parameters to the script, which override the ones in the JSON file. See `python -m tf_crnn.train -h`

### Contents
* `model.py` : definition of the model
* `data_handler.py` : functions for data loading, preprocessing and data augmentation
* `config.py` : `class Params` manages parameters of model and experiments
* `decoding.py` : helper fucntion to transform characters to words
* `train.py` : script to launch for training the model, more info on the parameters and options inside
* `export_model.py`: script to export a model once trained, i.e for serving
* Extra : `hlp/numbers_mnist_generator.py` : generates a sequence of digits to form a number using the MNIST database
* Extra : `hlp/csv_path_convertor.py` : converts a csv file with relative paths to a csv file with absolute paths



