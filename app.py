############################################# MODELLO 1: IMSyPP #############################################
from flask import Flask, request, render_template, redirect, url_for
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import os
import speech_recognition as sr  # Libreria per il riconoscimento vocale
from pydub import AudioSegment  # Libreria per la manipolazione audio
import requests

app = Flask(__name__)

# Percorso per salvare i file audio caricati (se necessario)
UPLOAD_FOLDER = 'uploads/audio_files/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Carica il tokenizer e il modello
tokenizer = AutoTokenizer.from_pretrained("IMSyPP/hate_speech_en")
model = AutoModelForSequenceClassification.from_pretrained("IMSyPP/hate_speech_en")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)


# Dizionario per mappare le etichette del modello con le etichette desiderate
label_mapping = {
    'LABEL_0': 'acceptable',
    'LABEL_1': 'inappropriate',
    'LABEL_2': 'offensive',
    'LABEL_3': 'violent'
}


def convert_audio_to_text(file_path):
    """Converte un file audio in testo utilizzando la libreria SpeechRecognition."""
    recognizer = sr.Recognizer()

    # Converti il file in formato .wav per una migliore compatibilità
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.rsplit(".", 1)[0] + ".wav"
    audio.export(wav_path, format="wav")

    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = "Audio non comprensibile."
        except sr.RequestError as e:
            text = f"Errore del servizio di riconoscimento vocale; {e}"

    # Rimuove il file wav temporaneo
    os.remove(wav_path)

    return text


def generate_explanation(comment, classification):
    # URL dell'API locale di LM Studio
    url = "http://localhost:8080/v1/completions"

    # Intestazioni della richiesta
    headers = {
        "Content-Type": "application/json"
    }

    # Prompt migliorato per dare contesto e istruzioni chiare al modello
    prompt = f"""
    Given the following comment: "{comment}"
    The comment has been classified as '{classification}'.

    Provide a detailed explanation of why this comment might be considered {classification}.
    Focus on the linguistic elements and the context of the comment that contribute to this classification.
    """

    # Corpo della richiesta
    data = {
        "prompt": prompt,
        "temperature": 0.5,  # temperatura ridotta per risposte meno creative e più mirate
        "max-tokens": 1000
    }

    try:
        # Invia una richiesta POST all'API di LM Studio
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()

        # Ottieni il risultato JSON dalla risposta
        result = response.json()

        # Estrai la spiegazione generata
        explanation = result.get('choices', [{}])[0].get('text', '')

        return explanation.strip()

    except requests.exceptions.HTTPError as http_err:
        print(f"Errore HTTP: {http_err}")
    except Exception as err:
        print(f"Errore: {err}")
    
    return None



@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_text = ""

        if "text" in request.form and request.form["text"].strip():
            input_text = request.form["text"]
        elif "audio" in request.files and request.files["audio"].filename != "":
            audio_file = request.files["audio"]
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
            audio_file.save(audio_path)
            input_text = convert_audio_to_text(audio_path)

        if input_text:
            results = classifier(input_text)
            
            for result in results:
                result['label'] = label_mapping.get(result['label'], result['label'])
                
                # Chiama la funzione di generazione con il commento e l'etichetta di classificazione
                result['explanation'] = generate_explanation(input_text, result['label'])

            return render_template("result.html", results=results, input_text=input_text)
        else:
            return redirect(url_for('home'))

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)


############################################# MODELLO 2: TWEETEVAL #############################################
'''
from flask import Flask, request, render_template, redirect, url_for
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import os
import speech_recognition as sr
from pydub import AudioSegment
import requests

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/audio_files/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Caricamento delle pipeline per diversi compiti
tokenizer_offensive = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
model_offensive = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-offensive")
classifier_offensive = pipeline("text-classification", model=model_offensive, tokenizer=tokenizer_offensive)

classifier_emotion = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-emotion")
classifier_hate_speech = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-hate")
classifier_irony = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-irony")

# Mappatura delle etichette del modello alle categorie desiderate
label_mapping_offensive = {
    'LABEL_0': 'acceptable',  # Non offensivo
    'LABEL_1': 'offensive'    # Offensivo
}

def convert_audio_to_text(file_path):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.rsplit(".", 1)[0] + ".wav"
    audio.export(wav_path, format="wav")

    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = "Audio non comprensibile."
        except sr.RequestError as e:
            text = f"Errore del servizio di riconoscimento vocale; {e}"

    os.remove(wav_path)
    return text

def generate_explanation(comment, classification):
    url = "http://localhost:8080/v1/completions"
    headers = {"Content-Type": "application/json"}
    prompt = f"""
    Given the following comment: "{comment}"
    The comment has been classified as '{classification}'.

    Provide a detailed explanation of why this comment might be considered {classification}.
    Focus on the linguistic elements and the context of the comment that contribute to this classification.
    """
    data = {
        "prompt": prompt,
        "temperature": 0.5,
        "max-tokens": 1000
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        explanation = result.get('choices', [{}])[0].get('text', '')
        return explanation.strip()

    except requests.exceptions.HTTPError as http_err:
        print(f"Errore HTTP: {http_err}")
    except Exception as err:
        print(f"Errore: {err}")
    
    return None

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_text = ""

        if "text" in request.form and request.form["text"].strip():
            input_text = request.form["text"]
        elif "audio" in request.files and request.files["audio"].filename != "":
            audio_file = request.files["audio"]
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
            audio_file.save(audio_path)
            input_text = convert_audio_to_text(audio_path)

        if input_text:
            # Esegui la classificazione utilizzando diverse pipeline
            results_offensive = classifier_offensive(input_text)
            results_emotion = classifier_emotion(input_text)
            results_hate_speech = classifier_hate_speech(input_text)
            results_irony = classifier_irony(input_text)
            
            # Mappatura delle etichette per il modello offensivo
            for result in results_offensive:
                result['label'] = label_mapping_offensive.get(result['label'], result['label'])
                result['explanation'] = generate_explanation(input_text, result['label'])

            # Raccolta di tutti i risultati
            results = {
                "offensive": results_offensive,
                "emotion": results_emotion,
                "hate_speech": results_hate_speech,
                "irony": results_irony
            }

            return render_template("result.html", results=results, input_text=input_text)
        else:
            return redirect(url_for('home'))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

'''

############################################# MODELLO 3: HateBERT #############################################
'''
from flask import Flask, request, render_template, redirect, url_for
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import os
import speech_recognition as sr
from pydub import AudioSegment
import requests
import re  # Libreria per identificare gruppi mirati

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/audio_files/'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Carica il modello HateBERT
tokenizer = AutoTokenizer.from_pretrained("GroNLP/hateBERT")
model = AutoModelForSequenceClassification.from_pretrained("GroNLP/hateBERT")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Dizionario per etichette di classificazione
label_mapping = {
    'LABEL_0': 'non-offensive',
    'LABEL_1': 'offensive',
    'LABEL_2': 'hate speech'
}

# Dizionario per i gruppi mirati
target_groups = {
    'women': 'genere',
    'migrants': 'nazionalità/etnia',
    'race': 'razza',
    'religion': 'religione',
    'sexuality': 'orientamento sessuale'
}

def convert_audio_to_text(file_path):
    """Converte un file audio in testo utilizzando la libreria SpeechRecognition."""
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_file(file_path)
    wav_path = file_path.rsplit(".", 1)[0] + ".wav"
    audio.export(wav_path, format="wav")

    with sr.AudioFile(wav_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
        except sr.UnknownValueError:
            text = "Audio non comprensibile."
        except sr.RequestError as e:
            text = f"Errore del servizio di riconoscimento vocale; {e}"

    os.remove(wav_path)
    return text

def identify_target_group(text):
    """Identifica il gruppo mirato se viene rilevato hate speech."""
    for group in target_groups:
        if re.search(group, text, re.IGNORECASE):
            return target_groups[group]
    return "non identificato"

def generate_explanation(comment, classification):
    # URL dell'API locale di LM Studio
    url = "http://localhost:8080/v1/completions"

    # Intestazioni della richiesta
    headers = {
        "Content-Type": "application/json"
    }

    # Prompt migliorato per dare contesto e istruzioni chiare al modello
    prompt = f"""
    Given the following comment: "{comment}"
    The comment has been classified as '{classification}'.

    Provide a detailed explanation of why this comment might be considered {classification}.
    Focus on the linguistic elements and the context of the comment that contribute to this classification.
    """

    # Corpo della richiesta
    data = {
        "prompt": prompt,
        "temperature": 0.5,  # Ridotta temperatura per risposte meno creative e più mirate
        "max-tokens": 1000
    }

    try:
        # Invia una richiesta POST all'API di LM Studio
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()

        # Ottieni il risultato JSON dalla risposta
        result = response.json()

        # Estrai la spiegazione generata
        explanation = result.get('choices', [{}])[0].get('text', '')

        return explanation.strip()

    except requests.exceptions.HTTPError as http_err:
        print(f"Errore HTTP: {http_err}")
    except Exception as err:
        print(f"Errore: {err}")
    
    return None

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_text = ""
        if "text" in request.form and request.form["text"].strip():
            input_text = request.form["text"]
        elif "audio" in request.files and request.files["audio"].filename != "":
            audio_file = request.files["audio"]
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], audio_file.filename)
            audio_file.save(audio_path)
            input_text = convert_audio_to_text(audio_path)

        if input_text:
            results = classifier(input_text)
            for result in results:
                result['label'] = label_mapping.get(result['label'], result['label'])
                if result['label'] == 'hate speech':
                    result['target_group'] = identify_target_group(input_text)
                result['explanation'] = generate_explanation(input_text, result['label'])
            return render_template("result.html", results=results, input_text=input_text)
        else:
            return redirect(url_for('home'))

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
'''