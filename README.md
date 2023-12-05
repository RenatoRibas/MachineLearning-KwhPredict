## Projeto: Pipeline completo de machine learning
# Previsão Sustentável de Consumo de Energia na Indústria Siderúrgica

<div align="center">
  <img src="Images/Logo.png" alt="Logo">
</div>

### Integrantes:
* [Renato Ribas Campos](https://github.com/RenatoRibas)



### Descrição do Projeto:

Importância crítica da previsão sustentável de consumo de energia na indústria siderúrgica.

Eficiência Operacional:
Permite que as siderúrgicas otimizem o uso de energia, maximizando a eficiência operacional.
Isso resulta em uma produção mais eficiente e redução de custos.

Sustentabilidade Ambiental:
Contribui para práticas mais sustentáveis, reduzindo a pegada de carbono e minimizando o impacto ambiental associado ao consumo excessivo e não otimizado de energia.

Planejamento Estratégico:
Capacita as empresas a realizar um planejamento estratégico mais preciso, antecipando variações na demanda de energia e permitindo a adoção de medidas proativas para lidar com essas flutuações


### Metodologia:

Este projeto adota o metodo de regressão para a previsão sustentável de consumo de energia na indústria siderúrgica. Os passos-chave abrangem desde a coleta de dados até o treinamento do modelo.

Coleta de Dados:
Utilizamos dados históricos abrangendo diversas variáveis, incluindo:
Reativo atual com atraso (kVarh)
Reativo atual com adiantamento (kVarh)
Emissões de CO2
Fator de potência com atraso atual
Fator de potência com adiantamento atual

Pré-processamento de Dados:
Realizamos a etapa de pré-processamento para limpeza e normalização dos dados.

Escolha do Modelo de Machine Learning:
Optamos por um modelo de machine learning regressão linear, devido à sua capacidade de lidar com relações complexas entre as variáveis de entrada.

Treinamento do Modelo:
O modelo foi treinado utilizando dados históricos, ajustando-se aos padrões identificados durante a fase de pré-processamento.


### Base de dados:

https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption

### Tecnologias, bibliotecas e frameworks:

import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split,GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor # KNN
from sklearn.svm import SVR

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


import seaborn as sns # Retirar outliers
import matplotlib.pyplot as plt