import os
import typer
import pandas as pd
import sklearn
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data import Xs, ys
from filters import bandpass, raw
from preprocess import pca, lasso
from models.base import ModelBase
from models import svm, dt, knn, multi_logistic
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import config

def main(output_dir):
    filters = [bandpass, raw]  
    preprocessers = [pca,lasso]
    models: List[ModelBase] = [svm.Model(), dt.Model(), multi_logistic.Model(), knn.Model()]

    bandpass_conf = config.filters["bandpass"]
    bandpass_conf["X"] = Xs


    for fit in filters:
        fit_name = fit.__name__.split('.')[-1]
        conf = config.filters[fit_name]
        match fit_name:
            case 'bandpass':
                conf['X'] = Xs
            case 'raw':
                conf['X'] = Xs
            
        Xs_filtered = fit.filter(**conf)
        # Xs_flat = Xs_filtered.reshape(Xs_filtered.shape[0], -1)

        X_train, X_test, y_train, y_test = train_test_split(Xs_filtered, ys, test_size=0.1, stratify=ys)

        for prep in preprocessers:
            prep_name = prep.__name__.split('.')[-1]
            conf = config.preprocessers[prep_name]
            match prep_name:
                case 'lasso':
                    conf['X'] = X_train
                    conf['y'] = y_train
                case 'pca':
                    conf['X'] = X_train
            
            selector = prep.process(**conf)


            X_train_selected = selector.transform(X_train)
            X_test_selected = selector.transform(X_test)

            X_train_selected_flat = X_train_selected.reshape(X_train_selected.shape[0], -1)
            X_test_selected_flat = X_test_selected.reshape(X_test_selected.shape[0], -1)

            for m in models:
                model_name = m.name()
                store_path = os.path.join(output_dir, fit_name, prep_name, model_name)
                if not os.path.exists(store_path):
                    os.makedirs(store_path) 
                
                m.train(X_train_selected_flat, y_train)
                m.predict_and_report(X_test_selected_flat, y_test)
                m.save(store_path)

                # model = m.get_model(X_train_selected_flat, y_train)
                # ypred = model.predict(X_test_selected_flat)

                # result = ''
                # result += 'accuracy: ' + str(accuracy_score(y_test, ypred))
                # result += '\n\n'
                # result += 'classification report: \n' + classification_report(y_test, ypred)

                # with open(os.path.join(store_path, 'result.txt'), 'w') as f:
                #     f.write(result)


if __name__ == '__main__':
    typer.run(main)