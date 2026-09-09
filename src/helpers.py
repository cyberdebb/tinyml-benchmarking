import matplotlib.pyplot as plt
import numpy as np
from keras import models
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, roc_auc_score, roc_curve
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import os

def mkdir_recursive(path):
  if path == "":
    return
  sub_path = os.path.dirname(path)
  if not os.path.exists(sub_path):
    mkdir_recursive(sub_path)
  if not os.path.exists(path):
    print("Creating directory " + path)
    os.mkdir(path)

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
    fig.savefig('results/confusionMatrix-' + feature + '.eps', format='eps', dpi=1000)
    return ax


def PR_ROC_curves(ytrue, ypred, classes, ypred_mat):
    ybool = ypred == ytrue
    f, ax = plt.subplots(3, 4, figsize=(10, 10))
    ax = [a for i in ax for a in i]

    e = -1
    for c in classes:
        idx1 = [n for n, x in enumerate(ytrue) if classes[x] == c]
        idx2 = [n for n, x in enumerate(ypred) if classes[x] == c]
        idx = idx1 + idx2
        if idx == []:
            continue
        bi_ytrue = ytrue[idx]
        bi_prob = ypred_mat[idx, :]
        bi_ybool = np.array(ybool[idx])
        bi_yscore = np.array([bi_prob[x][bi_ytrue[x]] for x in range(len(idx))])
        try:
            print("AUC for {}: {}".format(c, roc_auc_score(bi_ybool + 0, bi_yscore)))
            e += 1
        except ValueError:
            continue
        ppvs, senss, thresholds = precision_recall_curve(bi_ybool, bi_yscore)
        cax = ax[2 * e]
        cax.plot(ppvs, senss, lw=2, label="Model")
        cax.set_xlim(-0.008, 1.05)
        cax.set_ylim(0.0, 1.05)
        cax.set_title("Class {}".format(c))
        cax.set_xlabel('Sensitivity (Recall)')
        cax.set_ylabel('PPV (Precision)')
        cax.legend(loc=3)

        fpr, tpr, thresholds = roc_curve(bi_ybool, bi_yscore)
        cax2 = ax[2 * e + 1]
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
    model.load_weights(config.trained_model)
    
    ypred_mat = model.predict(Xval)

    print("yval.shape", yval.shape)

    ytrue = np.argmax(yval, axis=1)
    yscore = np.array([ypred_mat[x][ytrue[x]] for x in range(len(yval))])
    ypred = np.argmax(ypred_mat, axis=1)
    
    print(classification_report(ytrue, ypred))
    plot_confusion_matrix(ytrue, ypred, classes, feature=config.feature, normalize=False)
    print("F1 score:", f1_score(ytrue, ypred, average=None))
    PR_ROC_curves(ytrue, ypred, classes, ypred_mat)


def plot_beat(beat_array, class_id=None):
    """
    Plota um único batimento cardíaco segmentado.
    
    Parâmetros:
    beat_array: array 1D com o sinal do batimento
    class_id: (opcional) ID da classe (0 a 4) para exibir no título
    """
    # Mapeamento reverso para exibir o nome da classe legível
    class_names = {
        0: "Normal (N)",
        1: "Supraventricular (S)",
        2: "Ventricular (V)",
        3: "Fusão (F)",
        4: "Desconhecido (Q)"
    }
    
    plt.figure(figsize=(8, 4))
    
    # Plota o sinal
    plt.plot(beat_array, color='#1f77b4', linewidth=2)
    
    # Configura o título com base na classe fornecida
    title = "Batimento Segmentado"
    if class_id is not None and class_id in class_names:
        title += f" - {class_names[class_id]}"
        
    plt.title(title)
    plt.xlabel("Amostras da Janela")
    plt.ylabel("Amplitude Normalizada")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Desenha uma linha vertical no centro indicando onde está o pico R
    center = len(beat_array) // 2
    plt.axvline(x=center, color='red', linestyle='--', label='Pico R (Centro)', alpha=0.6)
    plt.legend()
    
    plt.tight_layout()
    plt.show()
