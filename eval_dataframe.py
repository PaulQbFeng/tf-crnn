import os
import cv2
import argparse
import unicodedata
import numpy as np
import pandas as pd
import tensorflow as tf

from glob import glob
from src.loader import PredictionModel


def normalize_text(text):
    """Remove accents and other stuff from text"""
    return ''.join((c for c in unicodedata.normalize('NFD', text) \
                    if unicodedata.category(c) != 'Mn'))


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def eval_model(model, dataframe, nb_paths):
    """
    Args:
        model: model loaded with PredictionModel
        dataframe: dataframe of (path, label, corpus, old). Annotations tsv file.
        nb_paths: how many paths to consider.
    """
    examples = dataframe.copy()

    col_prob, col_rawpred, col_preds, col_confidence = [], [], [], []

    for example in examples.itertuples():
        img = cv2.imread(example.path)
        if img is None:
            continue
        img = img[..., 0]
        img = 255 * (img > img.mean())

        predictions = model.predict(img[:,:,np.newaxis], [example.corpus])
        if nb_paths == 1:
            pred_texts = predictions['words'][0].decode('latin1')
            pred_confidence = np.squeeze(predictions['score'])
        else:
            pred_texts = np.squeeze(predictions['words'])
            pred_likelihood = np.round(np.squeeze(predictions['score']), 3)
            pred_confidence = softmax(pred_likelihood)

        col_preds.append(pred_texts)
        col_confidence.append(pred_confidence)
        col_rawpred.append(np.squeeze(predictions['raw_predictions']))
        col_prob.append(np.squeeze(predictions['prob']))

    array_preds = np.asarray(col_preds)
    array_confidence = np.asarray(col_confidence)

    if nb_paths == 1:
        examples['pred_1'] = col_preds
        examples['score_1'] = col_confidence
    else:
        for i in range(nb_paths):
            examples['pred_{}'.format(i+1)] = [array_preds[:, i][j].decode('latin1') for j in range(len(array_preds))]
            examples['confidence_{}'.format(i+1)] = array_confidence[:, i]

    rawpred_array = np.asarray(col_rawpred)
    logprob_array = np.asarray(col_prob)

    return examples, rawpred_array, logprob_array


def write_predictions_by_epoch(training_name, nb_paths):
    """
    Store the dataframe in .tsv format and raw predictions/log probabilities in numpy files.
    -> /notebooks/report_result/
    """

    source = '/notebooks/'  # docker -v mapping on /home/paul.
    examples = pd.read_csv(source + 'accident-annotations/result_100_10_types_latin1_byreport.tsv',
                           encoding='latin1', sep='\t')

    os.makedirs(source + 'report_result/{}/pred_by_epoch'.format(training_name), mode=0o767, exist_ok=True)

    model_epochs = sorted(glob('/mnt/nfs/data/paul/generative/{}/export/*'.format(training_name)),
                      key=lambda x: int(os.path.basename(x)))[:-1]  # sort Remove last one cause it's a duplicate

    for i,model in enumerate(model_epochs):
        with tf.Graph().as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)

            with_elastic_model = PredictionModel(model, sess)
            print("Predictions after {} epoch of training\n".format(i+1))
            example, rawpred, logprob = eval_model(with_elastic_model, examples, nb_paths)

            example.to_csv(source + 'report_result/{name}/pred_by_epoch/examples_{name}_epoch_{}.tsv'
                           .format(i+1, name=training_name), sep='\t', encoding='latin1',index=False)

            np.savez_compressed(source + "report_result/{name}/pred_by_epoch/examples_{name}_epoch_{}.npz"
                                .format(i+1, name=training_name), rawpred_array=rawpred, logprob_array=logprob)

            sess.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-nm', '--training_name', type=str, required=True, help='name of the training model')
    parser.add_argument('-np', '--nb_paths', type=str, required=True, help='number of beam search path')
    parser.add_argument('-g', '--gpu', type=str, required=True, help='name of the tfrecords filename (add train or valid in the name)')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    write_predictions_by_epoch(args.training_name, int(args.nb_paths))
