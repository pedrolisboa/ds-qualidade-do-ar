Nessa análise, os resultados estão sendo gerados com Python 3.5. Os arquivos estão no formato ipynb, e podem ser lidos com a aplicação Jupyter Notebook.

```
$ pip install --upgrade pip
$ pip install jupyter
```

A configuração pode ser feita dentro de um virtualenv. O pacote usado para criá-lo foi o virtualenvwrapper

```
$ pip install virtualenvwrapper
$ mkvirtualenv--python=python3.5 nome_venv
```

Para rodar a análise, deve-se instalar os pacotes listados no arquivo requirements.txt.


```
$ pip install -r requirements.txt
```

Deve-se também adicionar um kernel no ipython com o ambiente configurado.

```
$ ipython kernel install --user --name=nome_venv
```

Em seguida, basta rodar o jupyter no diretório do projeto e selecionar os notebooks.

```
$ jupyter notebook
```

Descrição:

O conjunto de dados contém 9358 resultados médios de 5 sensores químicos de um dispositivo multisensor (PTXX.SX). O dispositivo estava localizado a nível da rua, dentro de uma cidade significativamente poluída. Os dados foram registrados de março de 2004 a fevereiro de 2005 (um ano). Valores ausentes são marcados com o valor -200. A medida de outros sensores também está disponível e algumas podem ser redundantes. A variável chave a ser analisada é PT08.S1 (CO), concentração de CO na atmosfera.
Informação das colunas:
1.	Date (DD/MM/YYYY)
2.	Time (HH.MM.SS)
3.	PT08.S1 (CO) – Variável de predição
4.	Non Metanic HydroCarbons Concentration (mg/m^3)
5.	4 Benzene Concentration (mg/m^3)
6.	PT08.S2 (NMHC)
7.	NOx Concentration (ppb)
8.	PT08.S3 (NOx)
9.	8 NO2 Concentration (mg/m^3)
10.	PT08.S4 (NO2s)
11.	PT08.S5 (O3)
12.	Temperature (C)
13.	Relative Humidity (%)
14.	AH Absolute Humidity

Perguntas:
1.	Escolha uma estratégia de tratamento de valores faltantes e outliers e justifique sua escolha.
2.	Para as quartas-feiras, quais os horários de pico na cidade (de maior concentração de CO) ?
3.	Quais as variáveis mais correlacionadas com a variável de predição?
4.	Crie um modelo de regressão de PT08.S1 a partir das demais variáveis. Avalie usando as métricas que julgar pertinente para o problema.
5.	Pergunta bônus: como as estações do ano interferem nas variáveis/predição e qual sua proposta de solução?
Essas perguntas servem apenas para direcionar a análise. Sinta-se livre para surpreender.
