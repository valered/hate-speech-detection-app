############################################# MODELLO 1: IMSyPP #############################################
'''
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Carica il dataset precedentemente salvato con le etichette di HateBERT
df = pd.read_csv('dataset/HateBERTdataset.csv')

# Carica il tokenizer e il modello IMSyPP
tokenizer = AutoTokenizer.from_pretrained("IMSyPP/hate_speech_en")
model = AutoModelForSequenceClassification.from_pretrained("IMSyPP/hate_speech_en")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer, truncation=True)

# Dizionario per la mappatura delle etichette di IMSyPP
label_mapping_imspp = {
    'LABEL_0': 'acceptable',
    'LABEL_1': 'inappropriate',
    'LABEL_2': 'offensive',
    'LABEL_3': 'violent'
}

# Funzione per trasformare l'etichetta predetta in etichetta binaria per IMSyPP
def transform_label_imspp(label):
    if label == 'acceptable':
        return 0
    else:
        return 1

# Colonna per memorizzare le etichette predette dal modello IMSyPP
imspp_predicted_labels = []

# Analizza ogni frase nel dataset utilizzando il modello IMSyPP
for idx, sentence in enumerate(df['Content']):
    result = classifier(sentence)[0]  # Prende il primo risultato della classificazione
    predicted_label = label_mapping_imspp[result['label']]  # Converte l'etichetta IMSyPP in testo
    binary_label = transform_label_imspp(predicted_label)  # Converte l'etichetta in binario
    imspp_predicted_labels.append(binary_label)

    # Log per monitorare i progressi
    if idx % 100 == 0:  # Stampa ogni 100 righe processate
        print(f"Processed {idx+1} rows out of {len(df)}")

# Aggiungi la colonna 'IMSyPP label' al dataset
df['IMSyPP label'] = imspp_predicted_labels

# Salva il dataset aggiornato
df.to_csv('dataset/IMSyPPdataset.csv', index=False)

print("Predictions added and saved to IMSyPPdataset.csv")
'''


################### GRAFICI: IMSyPP ########################
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Carica il dataset
df = pd.read_csv('dataset/dataset.csv')

# Assicurati di utilizzare il nome corretto delle colonne
y_true = df['Label']  # 'Label' è la colonna reale (valore vero)
y_pred = df['IMSyPP label']  # 'IMSyPP label' è la colonna predetta (valore predetto)

# Crea la Matrice di Confusione
conf_matrix = confusion_matrix(y_true, y_pred)

# Configura la visualizzazione della Matrice di Confusione e salvala
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Non-Offensivo', 'Offensivo'], yticklabels=['Non-Offensivo', 'Offensivo'])
plt.xlabel('Etichette Predette')
plt.ylabel('Etichette Reali')
plt.title('Matrice di Confusione (IMSyPP)')
plt.savefig('grafici/IMSyPPconfusion_matrix.png')
plt.close()  # Chiude la figura invece di mostrarla

# Stampa i valori esatti della matrice di confusione
print(f"Matrice di Confusione:\n{conf_matrix}")

# Stampa dei valori esatti per le etichette reali e predette
real_counts = df['Label'].value_counts()
pred_counts = df['IMSyPP label'].value_counts()
print(f"Etichette Reali:\nNon-Offensivo: {real_counts[0]}, Offensivo: {real_counts[1]}")
print(f"Etichette Predette:\nNon-Offensivo: {pred_counts[0]}, Offensivo: {pred_counts[1]}")

# Analisi di falsi positivi e falsi negativi
# Falsi positivi: etichette reali non offensive (0) e predette offensive (1)
false_positives = df[(y_true == 0) & (y_pred == 1)]
# Falsi negativi: etichette reali offensive (1) e predette non offensive (0)
false_negatives = df[(y_true == 1) & (y_pred == 0)]

# Stampa i valori esatti di falsi positivi e falsi negativi
print(f"Falsi Positivi (reali non-offensivi, predetti offensivi): {false_positives.shape[0]}")
print(f"Falsi Negativi (reali offensivi, predetti non-offensivi): {false_negatives.shape[0]}")

# Calcola l'Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Calcola la Precision
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.4f}")

# Calcola il Recall
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.4f}")

# Calcola l'F1 Score
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")

# Calcolo delle probabilità predette (necessario per la curva ROC)
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Grafico della curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC (IMSyPP)')
plt.legend(loc="lower right")
plt.savefig('grafici/IMSyPP_ROC_AUC.png')
plt.close()

# Stampa le metriche aggiuntive
print(f"AUC: {roc_auc:.4f}")
'''

############################################# MODELLO 3: HateBERT #############################################
'''
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Carica il dataset (assumiamo che il file sia un CSV con le colonne 'Content' e 'Label')
df = pd.read_csv('dataset/newdataset.csv')

# Carica il modello HateBERT e il tokenizer
tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT")

# Crea una pipeline per la classificazione del testo
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Dizionario per la mappatura delle etichette di HateBERT
label_mapping = {
    'LABEL_0': 'non-offensive',
    'LABEL_1': 'offensive',
    'LABEL_2': 'hate speech'
}

# Funzione per trasformare l'etichetta predetta in etichetta binaria
def transform_label(label):
    if label == 'offensive':
        return 1
    else:
        return 0

# Colonna per memorizzare le etichette predette
predicted_labels = []


for idx, content in enumerate(df['Content']):
    results = classifier(content)
    # Log per monitorare i progressi
    if idx % 100 == 0:  # Stampa ogni 100 righe processate
        print(f"Processed {idx+1} rows out of {len(df)}")

# Analizza ogni frase nel dataset
for sentence in df['Content']:
    result = classifier(sentence)[0]  # Prende il primo risultato della classificazione
    predicted_label = label_mapping[result['label']]  # Converte l'etichetta HateBERT in testo
    binary_label = transform_label(predicted_label)  # Converte l'etichetta in binario
    predicted_labels.append(binary_label)

# Aggiungi la colonna 'Predicted label' al dataset
df['HateBERT label'] = predicted_labels

# Salva il dataset aggiornato
df.to_csv('dataset/dataset.csv', index=False)

print("Predictions added and saved to dataset_with_predictions.csv")
'''

################### GRAFICI: HateBERT ########################
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc


# Carica il dataset
df = pd.read_csv('dataset/dataset.csv')

# Assicurati di utilizzare il nome corretto delle colonne
y_true = df['Label']  # 'Label' è la colonna reale (valore vero)
y_pred = df['HateBERT label']  # 'HateBERT label' è la colonna predetta (valore predetto)

# Crea la Matrice di Confusione
conf_matrix = confusion_matrix(y_true, y_pred)

# Configura la visualizzazione della Matrice di Confusione e salvala
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Reds', xticklabels=['Non-Offensivo', 'Offensivo'], yticklabels=['Non-Offensivo', 'Offensivo'])
plt.xlabel('Etichette Predette')
plt.ylabel('Etichette Reali')
plt.title('Matrice di Confusione')
plt.savefig('grafici/HateBERTconfusion_matrix.png')
plt.close()  # Chiude la figura invece di mostrarla

# Stampa i valori esatti della matrice di confusione
print(f"Matrice di Confusione:\n{conf_matrix}")

# Stampa dei valori esatti per le etichette reali e predette
real_counts = df['Label'].value_counts()
pred_counts = df['HateBERT label'].value_counts()
print(f"Etichette Reali:\nNon-Offensivo: {real_counts[0]}, Offensivo: {real_counts[1]}")
print(f"Etichette Predette:\nNon-Offensivo: {pred_counts[0]}, Offensivo: {pred_counts[1]}")

# Analisi di falsi positivi e falsi negativi
# Falsi positivi: etichette reali non offensive (0) e predette offensive (1)
false_positives = df[(y_true == 0) & (y_pred == 1)]
# Falsi negativi: etichette reali offensive (1) e predette non offensive (0)
false_negatives = df[(y_true == 1) & (y_pred == 0)]

# Stampa i valori esatti di falsi positivi e falsi negativi
print(f"Falsi Positivi (reali non-offensivi, predetti offensivi): {false_positives.shape[0]}")
print(f"Falsi Negativi (reali offensivi, predetti non-offensivi): {false_negatives.shape[0]}")

# Calcola l'Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Calcola la Precision
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.4f}")

# Calcola il Recall
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.4f}")

# Calcola l'F1 Score
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")

# Calcolo delle probabilità predette (necessario per la curva ROC)
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Grafico della curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='red', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC (HateBERT)')
plt.legend(loc="lower right")
plt.savefig('grafici/HateBERT_ROC_AUC.png')
plt.close()

# Stampa le metriche aggiuntive
print(f"AUC: {roc_auc:.4f}")
'''

############################################# MODELLO 2: TWEETEVAL #############################################
'''
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Carica il dataset esistente
df = pd.read_csv('dataset/IMSyPPdataset.csv')

# Carica il tokenizer e il modello TWEETEVAL per la classificazione dei contenuti offensivi
tokenizer_offensive = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
model_offensive = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")

# Crea la pipeline di classificazione per il modello TWEETEVAL
classifier_offensive = pipeline("text-classification", model=model_offensive, tokenizer=tokenizer_offensive)

# Funzione per mappare le etichette in binario
def transform_label_offensive(label):
    if label == 'offensive':
        return 1
    else:
        return 0

# Colonna per memorizzare le etichette predette
predicted_labels_offensive = []

# Classificazione e aggiunta delle etichette predette al dataset
for idx, sentence in enumerate(df['Content']):
    # Aggiungi troncamento e imposta max_length a 512 token
    results = classifier_offensive(sentence, truncation=True, max_length=512)  
    label = results[0]['label']  # Prendi l'etichetta predetta (offensive o non-offensive)
    binary_label = transform_label_offensive(label)  # Converte l'etichetta in formato binario
    predicted_labels_offensive.append(binary_label)

    # Log per monitorare i progressi
    if idx % 100 == 0:
        print(f"Processed {idx+1} rows out of {len(df)}")

# Aggiungi la colonna 'TWEETEVAL label' al dataset
df['TWEETEVAL label'] = predicted_labels_offensive

# Salva il dataset aggiornato
df.to_csv('dataset/IMSyPPdataset_with_TWEETEVAL.csv', index=False)

print("Predictions added and saved to IMSyPPdataset_with_TWEETEVAL.csv")
'''

################### GRAFICI: TWEETEVAL ########################
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc

# Carica il dataset
df = pd.read_csv('dataset/dataset.csv')

# Assicurati di utilizzare il nome corretto delle colonne
y_true = df['Label']  # 'Label' è la colonna reale (valore vero)
y_pred = df['TWEETEVAL label']  # 'TWEETEVAL label' è la colonna predetta (valore predetto)

# Crea la Matrice di Confusione
conf_matrix = confusion_matrix(y_true, y_pred)

# Configura la visualizzazione della Matrice di Confusione e salvala
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlOrBr', xticklabels=['Non-Offensivo', 'Offensivo'], yticklabels=['Non-Offensivo', 'Offensivo'])
plt.xlabel('Etichette Predette')
plt.ylabel('Etichette Reali')
plt.title('Matrice di Confusione (TweetEval)')
plt.savefig('grafici/TWEETEVALconfusion_matrix.png')
plt.close()  # Chiude la figura invece di mostrarla

# Stampa i valori esatti della matrice di confusione
print(f"Matrice di Confusione:\n{conf_matrix}")

# Stampa dei valori esatti per le etichette reali e predette
real_counts = df['Label'].value_counts()
pred_counts = df['TWEETEVAL label'].value_counts()
print(f"Etichette Reali:\nNon-Offensivo: {real_counts[0]}, Offensivo: {real_counts[1]}")
print(f"Etichette Predette:\nNon-Offensivo: {pred_counts[0]}, Offensivo: {pred_counts[1]}")

# Analisi di falsi positivi e falsi negativi
# Falsi positivi: etichette reali non offensive (0) e predette offensive (1)
false_positives = df[(y_true == 0) & (y_pred == 1)]
# Falsi negativi: etichette reali offensive (1) e predette non offensive (0)
false_negatives = df[(y_true == 1) & (y_pred == 0)]

# Stampa i valori esatti di falsi positivi e falsi negativi
print(f"Falsi Positivi (reali non-offensivi, predetti offensivi): {false_positives.shape[0]}")
print(f"Falsi Negativi (reali offensivi, predetti non-offensivi): {false_negatives.shape[0]}")


# Calcola l'Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Calcola la Precision
precision = precision_score(y_true, y_pred)
print(f"Precision: {precision:.4f}")

# Calcola il Recall
recall = recall_score(y_true, y_pred)
print(f"Recall: {recall:.4f}")

# Calcola l'F1 Score
f1 = f1_score(y_true, y_pred)
print(f"F1 Score: {f1:.4f}")

# Calcolo delle probabilità predette (necessario per la curva ROC)
fpr, tpr, _ = roc_curve(y_true, y_pred)
roc_auc = auc(fpr, tpr)

# Grafico della curva ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='yellow', lw=2, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Curva ROC (TWEETEVAL)')
plt.legend(loc="lower right")
plt.savefig('grafici/TWEETEVAL_ROC_AUC.png')
plt.close()

# Stampa le metriche aggiuntive
print(f"AUC: {roc_auc:.4f}")
'''
####################### CONFRONTO ######################
'''
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Carica il dataset
df = pd.read_csv('dataset/dataset.csv')

# Concordanza/Discordanza tra i modelli
def analyze_model_agreement(df, model_labels):
    agreement_count = 0
    disagreement_count = 0
    
    for idx, row in df.iterrows():
        if len(set([row[label] for label in model_labels])) == 1:
            agreement_count += 1
        else:
            disagreement_count += 1
    
    print(f"Modelli concordano su {agreement_count} righe.")
    print(f"Modelli discordano su {disagreement_count} righe.")

# Verifica che le colonne siano presenti nel dataset
expected_columns = ['Content', 'Label', 'IMSyPP label', 'TWEETEVAL label', 'HateBERT label']
missing_columns = [col for col in expected_columns if col not in df.columns]

if missing_columns:
    print(f"Attenzione! Mancano le seguenti colonne nel dataset: {missing_columns}")
else:
    print("Tutte le colonne necessarie sono presenti nel dataset.")

# Esegui l'analisi di concordanza/discordanza
analyze_model_agreement(df, ['IMSyPP label', 'TWEETEVAL label', 'HateBERT label'])
'''