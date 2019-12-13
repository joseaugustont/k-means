import numpy as np
from numpy import array
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import shutil
import math
from IPython.display import display


zeis = ["Califon / Estação Velha", "Catingueira / Riacho do Bodocongó - Bairro das Cidades", "Invasão da Macaíba / Novo Horizonte", "Invasão de Santa Cruz", "Invasão do Alto Branco", "Invasão do Pelourinho", "Inva~sao do verdejante", "Invasão dos Brotos", "Três Irmãs", "Vila de Santa Cruz", "Novo Cruzeiro", "Catolé de Zé Ferreira", "Jardim Europa", "Invasão Ramadinha II", "pedregal", "Jeremias", "Nossa Senhora Aparecida"]
fsi = [0.44, 0.11, 0.25, 0.35, 0.02, 0.53, 0.30, 0.36, 0.25, 0.13, 0.53, 0.11, 0.11, 0.24, 0.42, 0.10, 0.25]
gsi = [0.42, 0.11, 0.25, 0.35, 0.02, 0.48, 0.27, 0.29, 0.25, 0.12, 0.46, 0.11, 0.11, 0.24, 0.40, 0.09, 0.23]
osr = [1.32, 8.47, 2.97, 1.85, 45.21, 0.98, 2.43, 1.98, 3.04, 6.75, 1.02, 8.27, 8.52, 3.09, 1.45, 9.01, 3.04]

structure = {'FSI':fsi,'GSI':gsi, 'OSR':osr}
df = pd.DataFrame(data=structure, columns=['FSI', 'GSI', 'OSR'])


kmeans = KMeans(n_clusters=3, max_iter=1000).fit(df)
label = kmeans.labels_

df2 = pd.DataFrame(data={'ID':zeis, 'label':label}, columns=['ID', 'label'])

print(df2)