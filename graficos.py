""" En este script realizamos los graficos pertinentes para
    la fase de Analisis de Datos.

    Para visualizar el grafico siguiente hay que cerrar el que se esta
    viendo.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

FILENAME = 'pima-indians-diabetes.csv'
data = pd.read_csv(FILENAME, header=None, names=['n_emb','conc_gluc','pres_art','piel_tric','2h_ins','imc','pedi_fun','edad','SALIDA'])

D=pd.get_dummies(data)
normD=(D-D.min())/(D.max()-D.min())

#Grafica de la Matriz de Correlacion
plt.matshow(normD.corr())
plt.xticks(range(len(normD.columns)), normD.columns, rotation = 'vertical')
plt.yticks(range(len(normD.columns)), normD.columns)
plt.colorbar()
plt.show()

#Grafica de la correlacion Concetración de Glucosa - Salida
corr_conc_gluc = sns.jointplot(data = normD, x = "conc_gluc", y = "SALIDA", kind='kde') 
corr_conc_gluc.annotate(stats.pearsonr)
plt.show()

#Grafica de la correlacion IMC - Salida
corr_conc_gluc = sns.jointplot(data = normD, x = "imc", y = "SALIDA", kind='kde') 
corr_conc_gluc.annotate(stats.pearsonr)
plt.show()

#Grafica de la correlacion IMC - Concentración de Glucosa
corr_conc_gluc = sns.jointplot(data = normD, x = "imc", y = "conc_gluc", kind='kde') 
corr_conc_gluc.annotate(stats.pearsonr)
plt.show()