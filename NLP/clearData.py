'''
import pandas as pd

# Carica il file CSV
file_path = "dataset/HateSpeechDataset.csv"
df = pd.read_csv(file_path)

# Elimina solo la colonna 'Content_int'
df.drop(columns=['Content_int'], inplace=True)

# Rimuovi metà delle istanze campionando il 70% del dataset
df = df.sample(frac=0.08, random_state=42)

# Salva il DataFrame pulito in un nuovo file CSV
output_file_path = "dataset/cleaned_dataset.csv"
df.to_csv(output_file_path, index=False)

print(f"DataFrame pulito e ridotto del 70% salvato in {output_file_path}.")
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Carica il dataset
df = pd.read_csv("dataset/cleaned_dataset.csv")

# Visualizza le prime righe per capire la struttura
print(df.head())

# Conteggio della distribuzione
class_distribution = df['Label'].value_counts()
print("Distribuzione delle classi:\n", class_distribution)

# Visualizza la distribuzione delle classi con un grafico a barre
sns.countplot(data=df, x='Label', palette="Set2")
plt.title("Distribuzione delle Classi")
plt.xlabel("Tipo di Linguaggio")
plt.ylabel("Conteggio")
plt.show()

