from __future__ import division, print_function

import os

import h5py
import numpy as np
import tensorflow as tf
from keras.saving import register_keras_serializable


@register_keras_serializable(package="custom")
def zeropad(x):
    y = tf.zeros_like(x)
    return tf.concat([x, y], axis=2)

@register_keras_serializable(package="custom")
def zeropad_output_shape(input_shape):
    shape = list(input_shape)
    assert len(shape) == 3
    shape[2] *= 2
    return tuple(shape)

def mkdir_recursive(path):
  if path == "":
    return
  sub_path = os.path.dirname(path)
  if not os.path.exists(sub_path):
    mkdir_recursive(sub_path)
  if not os.path.exists(path):
    print("Creating directory " + path)
    os.mkdir(path)

def loaddata(input_size, feature):
    mkdir_recursive('dataset')
    print("Loading training data...")
    with h5py.File('dataset/train.keras', 'r') as f:
        trainData = {key: f[key][...] for key in f.keys()}
        
    print("Loading training labels...")
    with h5py.File('dataset/trainlabel.keras', 'r') as f:
        testlabelData = {key: f[key][...] for key in f.keys()}
        
    print("Available features in training data:", list(trainData.keys()))
    print("Available features in label data:", list(testlabelData.keys()))
        
    X = np.float32(trainData[feature])
    y = np.float32(testlabelData[feature])
    print("Training shapes before shuffle - X:", X.shape, "y:", y.shape)
    print("Any NaN in X:", np.any(np.isnan(X)), "y:", np.any(np.isnan(y)))
    
    att = np.concatenate((X,y), axis=1)
    np.random.shuffle(att)
    X, y = att[:,:input_size], att[:, input_size:]
    print("Training shapes after shuffle - X:", X.shape, "y:", y.shape)
    print("Any NaN after shuffle - X:", np.any(np.isnan(X)), "y:", np.any(np.isnan(y)))
    
    print("Loading validation data...")
    with h5py.File('dataset/test.keras', 'r') as f:
        valData = {key: f[key][...] for key in f.keys()}
        
    print("Loading validation labels...")
    with h5py.File('dataset/testlabel.keras', 'r') as f:
        vallabelData = {key: f[key][...] for key in f.keys()}
        
    Xval = np.float32(valData[feature])
    yval = np.float32(vallabelData[feature])
    print("Validation shapes - Xval:", Xval.shape, "yval:", yval.shape)
    print("Any NaN in validation - Xval:", np.any(np.isnan(Xval)), "yval:", np.any(np.isnan(yval)))
    
    return (X, y, Xval, yval)

def plot_confusion_matrix(y_true, y_pred, classes, feature,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    mkdir_recursive('results')
    fig.savefig('results/confusionMatrix-'+feature+'.eps', format='eps', dpi=1000)
    return ax

def PR_ROC_curves(ytrue, ypred, classes, ypred_mat):
    ybool = ypred == ytrue
    f, ax = plt.subplots(3,4,figsize=(10, 10))
    ax = [a for i in ax for a in i]

    e = -1
    for c in classes:
        idx1 = [n for n,x in enumerate(ytrue) if classes[x]==c]
        idx2 = [n for n,x in enumerate(ypred) if classes[x]==c]
        idx = idx1+idx2
        if idx == []:
            continue
        bi_ytrue = ytrue[idx]
        bi_prob = ypred_mat[idx, :]
        bi_ybool = np.array(ybool[idx])
        bi_yscore = np.array([bi_prob[x][bi_ytrue[x]] for x in range(len(idx))])
        try:
            print("AUC for {}: {}".format(c, roc_auc_score(bi_ybool+0, bi_yscore)))
            e+=1
        except ValueError:
            continue
        ppvs, senss, thresholds = precision_recall_curve(bi_ybool, bi_yscore)
        cax = ax[2*e]
        cax.plot(ppvs, senss, lw=2, label="Model")
        cax.set_xlim(-0.008, 1.05)
        cax.set_ylim(0.0, 1.05)
        cax.set_title("Class {}".format(c))
        cax.set_xlabel('Sensitivity (Recall)')
        cax.set_ylabel('PPV (Precision)')
        cax.legend(loc=3)

        fpr, tpr, thresholds = roc_curve(bi_ybool, bi_yscore)
        cax2 = ax[2*e+1]
        cax2.plot(fpr, tpr, lw=2, label="Model")
        cax2.set_xlim(-0.1, 1.)
        cax2.set_ylim(0.0, 1.05)
        cax2.set_title("Class {}".format(c))
        cax2.set_xlabel('1 - Specificity')
        cax2.set_ylabel('Sensitivity')
        cax2.legend(loc=4)

    mkdir_recursive("results")
    plt.savefig("results/model_prec_recall_and_roc.eps",
        dpi=400,
        format='eps',
        bbox_inches='tight')
    plt.close()

def print_results(config, model, Xval, yval, classes):
    model2 = model
    if config.trained_model:
        model.load_weights(config.trained_model)
    else:    
        model.load_weights('models/{}-latest.keras'.format(config.feature))
    # to combine different trained models. On testing  
    if config.ensemble:
        model2.load_weight('models/weights-V1.keras')
        ypred_mat = (model.predict(Xval) + model2.predict(Xval))/2
    else:
        ypred_mat = model.predict(Xval)  

    print("yval.shape",yval)

    ytrue = np.argmax(yval,axis=1)
    yscore = np.array([ypred_mat[x][ytrue[x]] for x in range(len(yval))])
    ypred = np.argmax(ypred_mat, axis=1)
    print(classification_report(ytrue, ypred))
    plot_confusion_matrix(ytrue, ypred, classes, feature=config.feature, normalize=False)
    print("F1 score:", f1_score(ytrue, ypred, average=None))
    PR_ROC_curves(ytrue, ypred, classes, ypred_mat)