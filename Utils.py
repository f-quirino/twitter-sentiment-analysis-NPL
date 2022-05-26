#Bibliotecas de manipulação de dados e pre processamento
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from unidecode import unidecode 
from wordcloud import WordCloud
from PIL import Image

#from spellchecker import SpellChecker (tentei usar essa lib mas não foi efetiva)
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.models import Word2Vec,doc2vec


# Bibliotecas para métricas de avaliação de modelos
from sklearn.model_selection import train_test_split,KFold,cross_validate,cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score,classification_report, plot_confusion_matrix, confusion_matrix

# Bibliotecas com os modelos de classificação
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def pre_processamento(linha, stopwords):
    '''
    Função que recebe cada linha do dataset e irá retirar menções, links, deixar tudo com letra minúscula,
    remover sorrisos, dígitos, acentuação, caracteres especiais, repetição de caracteres, espaços,
    tokenização, espaços vazios e retirar as stopwords. Utilização da biblioteca NLTK.
    '''
    
    #Removendo menções
    linha = re.sub(r"@[a-zA-Z0-9_]{1,50}", "", linha)
    
    #Removendo o links:
    linha = re.sub(r"(https:\/\/.+)", "", linha)
    
    #Colocando todas as palavras em minúsculo
    linha = linha.lower()
    
    #Removendo todos os sorrisos :p e :d
    linha = linha.replace(":d", "")
    linha = linha.replace(":p", "")
    
    #Removendo os digitos
    linha = re.sub(r"\d+", "", linha)
    
    #Removendo a acentuação
    linha = unidecode(linha)
    
    #Removendo os caracteres especiais
    linha = re.sub(r"[^a-zA-Z0-9]", " ", linha)
    
    #Removendo caracteres repetidos 3 ou mais vezes
    linha = re.sub(r"(\w)\1(\1+)", r"\1", linha)
    
    #Removendo os espaços do inicio e final de cada frase
    linha = linha.strip()
    
    #Tokenizando
    palavras = word_tokenize(linha)
    
    #Removendo as stopwords:
    linha = [palavra for palavra in palavras if palavra not in stopwords]
    
    #Removendo elementos vazio 
    linha = [palavra for palavra in linha if palavra != ""]
    
    return linha

# Criando uma função para gerar nossos datasets de wordcloud
def wc_df(df, column, filter=False, filter_type=None):
    '''
    Função que cria um dicionário com a quantidade de ocorrências de cada palavra no dataset
    '''

    # Se houver filtro usa ele
    if filter:
        
        # Criando um dataframe com as palavras únicas
        vocabulary = pd.DataFrame(
                                np.concatenate(df.loc[filter_type, column].values),
                                columns=["word"]
                                )
    else:
        # Criando um dataframe com as palavras únicas
        vocabulary = pd.DataFrame(
                                np.concatenate(df[column].values),
                                columns=["word"]
                                )
        
    # Criando um dicionário com as palavras únicas                            
    dict_wc = vocabulary.groupby("word").size().to_dict()
    return dict_wc

#Função de plot da word cloud
def plot_cloud(wordcloud):
    plt.figure(figsize=(20, 15))
    plt.imshow(wordcloud) 
    plt.axis("off")

# Colocar o train_test_splot
def baseline_transform(models, X, y, verbose=False):
    '''
    Função recebe uma lista de modelos, aplica uma transformação de bag of words no dataset e faz uma validação cruzada de 5 folds com cada modelo.
    Além disso ela calcula diversas métricas de avaliação para cada modelo e retorna um dataframe com essas métricas.
    Verbose foi utilizado para imprimir para dar alguma ideia ao usuário do progresso.
    '''
    
    # Dicionário de métricas de avaliação para cada modelo
    models_dict = {}
    # Lista de métricas de avaliação que usaremos
    metrics = ["recall_weighted", "precision_weighted", "f1_weighted", "accuracy", "roc_auc_ovr"]
    
    if verbose:
        print("Iniciando o loop de modelos")
    # Loop para cada modelo
    for model in models:
        if verbose:
            print(f"Fit do modelo {model} iniciado")
        # Criando o pipeline
        pipe = Pipeline(steps=[("vect", CountVectorizer()),
                                (model, models[model])])
        # Fazendo a validação cruzada
        scores = cross_validate(pipe, X, y, cv=5, scoring=metrics, return_train_score=True, n_jobs=-1)

        # Dicionário com as métricas de cada modelo                    
        models_dict[model] = {"recall_test": scores["test_recall_weighted"].mean(),
                                    "recall_train": scores["train_recall_weighted"].mean(),
                                    "precision_test": scores["test_precision_weighted"].mean(),
                                    "precision_train": scores["train_precision_weighted"].mean(),
                                    "f1_test": scores["test_f1_weighted"].mean(),
                                    "f1_train": scores["train_f1_weighted"].mean(),
                                    "accuracy_test": scores["test_accuracy"].mean(),
                                    "accuracy_train": scores["train_accuracy"].mean(),
                                    "roc_auc_test": scores["test_roc_auc_ovr"].mean(),
                                    "roc_auc_train": scores["train_roc_auc_ovr"].mean(),
                                    }
        if verbose:
            print(f"Fit do modelo {model} finalizado")                    
    tabela = pd.DataFrame(models_dict)
    return tabela

def train_test_word2vec(x,model):
    '''
    Função para criar as amostras de teste e treino do word2vec
    Recebe o dataset e o modelo word2vec
    '''
    X_w2v_sn = [] # lista para somar os vetores
    for phrase in x:
        vecs = []
        for word in phrase:
            if word in model.wv.index_to_key:
                vecs.append(model.wv.get_vector(word))
        if vecs:
            soma_normalizada = np.sum(vecs, axis=0) / np.linalg.norm(np.sum(vecs, axis=0))
            X_w2v_sn.append(soma_normalizada)
        else:
            X_w2v_sn.append(np.zeros(model.vector_size))
    return np.array(X_w2v_sn)

def read_corpus_doc2vec(list_sentences, tokens_only=False):
    '''
    Função para criar os identificadores únicos para os documentos
    '''
    if tokens_only:
        return list_sentences
    else:
        lista = []
        for i, line in enumerate(list_sentences):
            lista.append(doc2vec.TaggedDocument(line, [i]))
        return lista

def train_test_doc2vec(x, model):
    '''
    Função para criar as amostras de teste e treino do doc2vec
    '''
    X_d2v = []
    for phrase in x:
        X_d2v.append(model.infer_vector(phrase))
    return np.array(X_d2v)

def try_transformers(transformers_dict, model):
    '''
    Função para testar todas as transformações no dataset usando a regressão logística como modelo padrão.
    A função recebe um dicionário de datasets de treino e teste associados a cada transformação e retorna uma tabela de métricas de avaliação de cada transformação
    '''
    
    # Dicionário de métricas para cada transformador
    transformers_metrics = {}
    
    # Lista de métricas a serem avaliadas
    metrics = ["recall_weighted", "precision_weighted", "f1_weighted", "accuracy", "roc_auc_ovr"]

    # Loop para testar todas as transformações
    for transformer,train_test_sets in transformers_dict.items():
        print(transformer.upper())
        print(f"Treinando {transformer}")
        
        # Fazendo a validação cruzada (usando somente a base de treino)
        scores = cross_validate(model, train_test_sets["X_train"],
                                train_test_sets["y_train"], cv=5,
                                scoring=metrics, n_jobs=-1)
        print(f"Treinamento com {transformer} finalizado")
        print(f"Testando {transformer}")

        # Fitando o modelo agora com toda a base de treino
        model.fit(train_test_sets["X_train"], train_test_sets["y_train"])
        
        # Predizendo com a base de teste 
        y_pred_test = model.predict(train_test_sets["X_test"])
        
        # Predizendo a probabilidade de cada label
        y_pred_test_prob = model.predict_proba(train_test_sets["X_test"])

        # Calculando as métricas de avaliação do teste
        accuracy_test = accuracy_score(train_test_sets["y_test"], y_pred_test)
        precision_test = precision_score(train_test_sets["y_test"], y_pred_test, average="weighted")
        recall_test = recall_score(train_test_sets["y_test"], y_pred_test, average="weighted")
        f1_test = f1_score(train_test_sets["y_test"], y_pred_test, average="weighted")
        auc_test = roc_auc_score(train_test_sets["y_test"], y_pred_test_prob,multi_class="ovr")
                
        # Dicionário com as métricas de cada modelo                    
        transformers_metrics[transformer] = {"recall_test": recall_test,
                                    "recall_train": scores["test_recall_weighted"].mean(),
                                    "precision_test": precision_test,
                                    "precision_train": scores["test_precision_weighted"].mean(),
                                    "f1_test": f1_test,
                                    "f1_train": scores["test_f1_weighted"].mean(),
                                    "accuracy_test": accuracy_test,
                                    "accuracy_train": scores["test_accuracy"].mean(),
                                    "roc_auc_test": auc_test,
                                    "roc_auc_train": scores["test_roc_auc_ovr"].mean(),
                                    }
        print(f"Teste com {transformer} finalizado")

        print('''        
                ###############################################
                ############ Classification Report ############
                ###############################################
            ''' )   
        print(classification_report(train_test_sets["y_test"] ,y_pred_test))
    
        print(50*"*", '\n')
        
    table = pd.DataFrame(transformers_metrics)
    return table