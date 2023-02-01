from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np
import os


def train_SVM(features,
              labels,
              feature_use = ['spectral_centroid','spectral_bandwidth'],
              valid_ratio = 0.25,
              kernel_svm = 'rbf',
              gamma_smv = 0.5,
              C_svm = 1,
              path_main = "./Apprentissage/A000_SVM/",
              plot_title = "SVM pour la classification des tables",
              model_name = "SVM",
              writer = None):
    
    best_trained_model_path = "{}/runs/{}/{}_best.pt".format(path_main, model_name,model_name)




    # Compute de X and y
    for i,feature in enumerate(feature_use):
        if feature not in ['rms','spectral_centroid','spectral_bandwidth','spectral_rolloff','spectral_flatness','zero_crossing_rate']:
            print("Feature {} not in the list of available features".format(feature))
            return
        else:
            if i == 0:
                X = features[feature]
            else:
                X = np.concatenate((X,features[feature]),axis=-1)

    y = labels

    # Diviser les données en données d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = valid_ratio)



    # Entraîner le modèle SVM
    clf = svm.SVC(gamma = gamma_smv,
                  C = C_svm, 
                  kernel = kernel_svm)
    clf.fit(X_train, y_train)

    # Évaluer le modèle
    accuracy = clf.score(X_test, y_test)
    print("Accuracy: ", accuracy)
    writer.add_scalar("Accuracy", accuracy)
    writer.flush()

    # Tracer les données de classification et les frontières de décision
    fig = plt.figure()
    plot_decision_regions(X, y, clf=clf, legend=2)

    # Ajouter les labels d'axes et le titre
    plt.xlabel(feature_use[0])
    plt.ylabel(feature_use[1])
    plt.title(plot_title)

    writer.add_figure("Classification", fig)
    writer.flush()
    

    # Sauvegarde du graphique dans le dossier Figures au format pdf
    #config_path = "{}/runs/{}".format(path_main, model_name)
    #os.mkdir(config_path+"/Figures")
    # Afficher le graphique
    #plt.savefig(config_path+"/Figures/"+model_name+".pdf", dpi = 300, bbox_inches = "tight", format = 'pdf')


    return clf, accuracy