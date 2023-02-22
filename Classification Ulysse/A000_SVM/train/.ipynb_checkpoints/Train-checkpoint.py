from sklearn import svm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions
import numpy as np
import os
import pickle
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler

class train():
    def __init__(self, 
                 features,
                 labels,
                 classes_names,
                 feature_use = ['spectral_centroid','spectral_bandwidth'],
                 valid_ratio = 0.25,
                 kernel_svm = 'rbf',
                 C_svm = 1,
                 path_main = "./Apprentissage/A000_SVM/",
                 plot_title = "SVM pour la classification des tables",
                 model_name = "SVM",
                 writer = None,
                 step = 10,
                 n_features = 2
                 ):
        super(train, self).__init__()
        # input parameters
        self.features = features
        self.n_features = n_features
        self.labels = labels
        self.classes_names = classes_names
        self.feature_use = feature_use
        self.model_name = model_name
        self.path_main = path_main
        self.best_trained_model_path = "{}/runs/{}/{}_best.pickle".format(self.path_main, self.model_name,self.model_name)
        self.best_model_path = "{}/runs/{}/{}_".format(self.path_main, self.model_name,self.model_name)
        # Save parameters
        self.writer = writer
        self.plot_title = plot_title

        self.valid_ratio = valid_ratio
        # SVM parameters
        self.kernel_svm = kernel_svm
        self.C_svm = C_svm
        self.step = step


    def train_step(self):
        if self.n_features == 2:
            # Compute de X and y
            v = 0
            Tot=[]
            
            y = self.labels
            for k,feature in enumerate(self.feature_use):
                if feature not in ['rms','spectral_centroid','spectral_bandwidth','spectral_rolloff','spectral_flatness','zero_crossing_rate']:
                    print("Feature {} not in the list of available features".format(feature))
                    return
                X_temp = []
                X_temp.append(self.features[feature])
                for h,feature2 in enumerate(self.feature_use):
                    
                    if h>k or len(X_temp)<self.n_features :
                        if h<=k:
                            continue
                        else:
                            v+=1
                            X_temp.append(self.features[feature2])
                            X = np.array(X_temp).T
                            Tot.append((feature,feature2))
                            

                            # Diviser les données en données d'entraînement et de test
                            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.valid_ratio, random_state=42)
            
                            for i,c in enumerate(np.linspace(self.C_svm[0],self.C_svm[-1],self.step)):
                                #print("Iteration : ",i)
                            
                                # Entraîner le modèle SVM
                                
                                clf = svm.SVC(gamma = 'scale', # gamma = gamma,
                                                C = c, 
                                                kernel = self.kernel_svm)
                                gamma =  1 / ((k+1) * X_train.var())
                                clf.fit(X_train, y_train)
                                
                                # Évaluer le modèle
                                accuracy = clf.score(X_test, y_test)
                                

                                # Save the best model
                                if i == 0 :
                                    best_accuracy = accuracy
                                    best_accuracy_c = c
                                    best_accuracy_gamma = gamma
                                    best_accuracy_model = clf
                                    
                                    
                                else:
                                    if accuracy > best_accuracy:
                                        best_accuracy = accuracy
                                        best_accuracy_c = c
                                        best_accuracy_gamma = gamma
                                        best_accuracy_model = clf           
                            pickle.dump(best_accuracy_model, open("{}{}_{}_best_model.pickle".format(self.best_model_path,feature,feature2), "wb"))
                            
                            ####### Visualisation des résultats #######
                            
                            self.writer.add_scalar("Accuracy", accuracy,v)
                            self.writer.add_text('Liste des configs', str(Tot))
                            self.writer.flush()
                            # Tracer les données de classification et les frontières de décision
                            fig = plt.figure()
                            ax = plot_decision_regions(X, 
                                                    y, 
                                                    clf=best_accuracy_model, 
                                                    legend=2,
                                                    zoom_factor=2)
                            handles, labels = ax.get_legend_handles_labels()
                            ax.legend(handles, self.classes_names,framealpha=0.9, scatterpoints=1)
                            plt.title(self.plot_title+"\nc = {}, gamma = {} with \n{} and {}".format(best_accuracy_c,best_accuracy_gamma,feature,feature2))
                            plt.xlabel(feature)
                            plt.ylabel(feature2)
                            #plt.show()
                            self.writer.add_figure("Visualisation", fig,v)
                            self.writer.flush()
                            # Plot the best model for acombinaison of features        
                            disp = ConfusionMatrixDisplay.from_estimator(
                                best_accuracy_model,
                                X_test,
                                y_test,
                                display_labels=self.classes_names,
                                cmap=plt.cm.Blues,
                                normalize=None,
                            )
                            disp.ax_.set_title("Confusion Matrix  for features : \n{} and {}".format(feature,feature2))
                            
                            self.writer.add_figure("Confusion Matrix", disp.figure_,v)
                            self.writer.flush()

                            print("Best accuracy : {} with C = {} and gamma = {} for the following features : {} and {}".format(best_accuracy,best_accuracy_c,best_accuracy_gamma,feature,feature2))
                            
                            
                            ###### Save best global model ######
                            if v == 1 :
                                global_best_accuracy = best_accuracy
                                global_best_accuracy_c = best_accuracy_c
                                global_best_accuracy_gamma = best_accuracy_gamma
                                global_best_accuracy_model = best_accuracy_model
                                global_best_accuracy_features = (feature,feature2)
                                
                                
                            else:
                                if accuracy > best_accuracy:
                                    global_best_accuracy = best_accuracy
                                    global_best_accuracy_c = best_accuracy_c
                                    global_best_accuracy_gamma = best_accuracy_gamma
                                    global_best_accuracy_model = best_accuracy_model
                                    global_best_accuracy_features = (feature,feature2)


                            ###### Reset de X ######
                            X_temp = []
                            X_temp.append(self.features[feature])

            ###### Save best global model ######
            pickle.dump(global_best_accuracy_model, open("{}{}_{}_global_best_model.pickle".format(self.best_model_path,global_best_accuracy_features[0],global_best_accuracy_features[1]), "wb"))
            print("Global best accuracy : {} with C = {} and gamma = {} for the following features : {} and {}".format(global_best_accuracy,global_best_accuracy_c,global_best_accuracy_gamma,global_best_accuracy_features[0],global_best_accuracy_features[1]))
        
            
        else :
            # Compute de X and y
            X= []
            for k,feature in enumerate(self.feature_use):
                if feature not in ['rms','spectral_centroid','spectral_bandwidth','spectral_rolloff','spectral_flatness','zero_crossing_rate']:
                    print("Feature {} not in the list of available features".format(feature))
                    return
                else:
                    X.append(self.features[feature])
            X=np.asarray(X).T
            
            y = self.labels

            # Diviser les données en données d'entraînement et de test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = self.valid_ratio, random_state=42)
            
            for i,c in enumerate(np.linspace(self.C_svm[0],self.C_svm[-1],self.step)):
                # Entraîner le modèle SVM
                clf = svm.SVC(gamma = 'scale', # gamma = gamma,
                                C = c, 
                                kernel = self.kernel_svm)
                gamma =  1 / ((k+1) * X_train.var())
                clf.fit(X_train, y_train)
                
                # Évaluer le modèle
                accuracy = clf.score(X_test, y_test)
                #print("Accuracy: ", accuracy)
                self.writer.add_scalar("Accuracy", accuracy,i)
                self.writer.flush()
                
                disp = ConfusionMatrixDisplay.from_estimator(
                    clf,
                    X_test,
                    y_test,
                    display_labels=self.classes_names,
                    cmap=plt.cm.Blues,
                    normalize=None,
                )
                disp.ax_.set_title("Confusion Matrix for C = {}".format(c))
                #plt.show()
                self.writer.add_figure("Confusion Matrix", disp.figure_,i)
                self.writer.flush()

                

                # Save the best model
                if i == 0 :
                    best_accuracy = accuracy
                    best_accuracy_c = c
                    best_accuracy_gamma = gamma
                    best_accuracy_model = clf
                    
                else:
                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_accuracy_c = c
                        best_accuracy_gamma = gamma
                        best_accuracy_model = clf
                        
            pickle.dump(best_accuracy_model, open(self.best_trained_model_path, "wb"))
            print("Best accuracy : {} with C = {} and gamma = {} for the following features{}".format(best_accuracy,best_accuracy_c,best_accuracy_gamma,self.feature_use))
            

        
        



"""         for i,c in enumerate(np.linspace(self.C_svm[0],self.C_svm[-1],self.step)):
            print("Iteration : ",i)
          
            # Entraîner le modèle SVM
            
            clf = svm.SVC(gamma = 'scale', # gamma = gamma,
                            C = c, 
                            kernel = self.kernel_svm)
            gamma =  1 / ((k+1) * X_train.var())
            clf.fit(X_train, y_train)
            
            # Évaluer le modèle
            accuracy = clf.score(X_test, y_test)
            self.writer.add_scalar("Accuracy", accuracy,i)
            self.writer.flush()

            # Save the best model
            if i == 0 :
                best_accuracy = accuracy
                best_accuracy_c = c
                best_accuracy_gamma = gamma
                best_accuracy_model = clf
                pickle.dump(best_accuracy_model, open(self.best_trained_model_path, "wb"))
                
            else:
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_accuracy_c = c
                    best_accuracy_gamma = gamma
                    best_accuracy_model = clf
                    pickle.dump(best_accuracy_model, open(self.best_trained_model_path, "wb"))
        
                    
        # Plot the best model for acombinaison of features        
        disp = ConfusionMatrixDisplay.from_estimator(
            best_accuracy_model,
            X_test,
            y_test,
            display_labels=self.classes_names,
            cmap=plt.cm.Blues,
            normalize=None,
        )
        disp.ax_.set_title("Confusion Matrix Best Model")
        
        self.writer.add_figure("Confusion Matrix", disp.figure_,i)
        self.writer.flush()

        print("Best accuracy : {} with C = {} and gamma = {} for the following features{}".format(best_accuracy,best_accuracy_c,best_accuracy_gamma,self.feature_use)) """