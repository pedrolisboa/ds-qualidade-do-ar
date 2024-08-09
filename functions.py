import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from keras.models import Sequential
from keras.layers import Dense

def get_na_sequences(dataset, variable): 
    """Obtém o tamanho de cada sequência de valores faltantes para uma variável no dataset""" 
    missing_seq_lengths = list()
    for group in itertools.groupby(dataset[variable].isna(), lambda x: True if x else False):
        missing_seq_lengths.append(sum(list(group[1])))
    return list(filter(lambda x: True if x!=0 else False, missing_seq_lengths))


def plot_daily_data(dataset, columns, folder_name):
    """
    Gera gráficos da evolução das variáveis ao longo dos dias do ano separadas por hora. 
    
    Args:
        dataset: dataframe sendo analisado
        columns: array com o nome das colunas do dataframe contendo as variáveis a serem analisadas
        folder_name: caminho dentro do diretório pictures onde os gráficos serão salvos.
                     Caso o caminho não exista, será criado 
    """
    for hour, hour_df in dataset.groupby("Hour"):
        na_cols = hour_df[columns]
        fig, axes = plt.subplots(nrows=len(columns), figsize=(25,10))
        for ax, column in zip(axes, na_cols):
    #         sns.lineplot(x="Date", y=column, data=hour_df, ax=ax)
            ax.plot(hour_df["Date"], na_cols[column])
            ax.xaxis.set_ticks(hour_df["Date"][::30])
            ax.set_title(column)
    #         axes.tick_params(axis='x', rotation=45)
        plt.subplots_adjust(hspace=1.0)
        filepath = "Pictures/" + folder_name 
        if not os.path.exists(filepath):
            os.mkdir(filepath)
        fig.savefig(filepath + "/Hour_%s.pdf" % hour)
        plt.close()
        
def error_measure(x, target, na_indices):
    reconstruction_err_df = abs(x.sub(target).iloc[na_indices])/target.iloc[na_indices]
    
    reconstruction_err_df.drop("Hour", axis=1, inplace=True)
    return reconstruction_err_df

def single_impute_na(subset,  target, method, na_indices, byhour=True):
    if method in ["mean", "mode", "median"]:
        method_fn = getattr(subset, method)
        x = subset.fillna(method_fn())
    else:
        if byhour:
            x = subset.groupby("Hour").apply(lambda x: x.interpolate(method=method))
        else:
            x = subset.interpolate(method=method)
    
    err_df = error_measure(x, target, na_indices)
    err_df["Method"] = str(method)
    return err_df

def iter_impute_na(subset, target, estimator, na_indices, byhour=True, max_iter=10):
    def impute_group(group):
        x = imp.fit_transform(group)
        x = pd.DataFrame(x, index=group.index, columns=group.columns)
        return x

    imp = IterativeImputer(estimator=estimator, missing_values=np.nan, max_iter=max_iter)
    if byhour:
        x = subset.groupby("Hour").apply(lambda x: impute_group(x))
    else:
        x = pd.DataFrame(imp.fit_transform(subset),
                         index=subset.index, columns=subset.columns)

    err_df = error_measure(x, target, na_indices)
    err_df["Method"] = type(estimator).__name__
    return err_df
