Hate Speech Detection App
Introduction
This project is part of a university course, aimed at testing three models for detecting harmful, offensive, or violent content on major social media platforms. The three models analyzed are HateBERT, IMSyPP, and TWEETEVAL, all available via Hugging Face.

In addition to these, a Large Language Model (LLM) is used to provide the user with an explanation as to why specific content has been flagged as toxic. The LLM used in this project is mradermacher/llama-3-8b-gpt-4o-GGUF, also accessible from Hugging Face.

Installation Requirements
To install and run the application, the following libraries and dependencies need to be installed:

Flask: A lightweight web framework to run the app server.
Transformers: For handling the NLP models.
Torch: For deep learning operations.
PyDub: For audio file processing, which might be required based on specific functionalities.
SpeechRecognition: To enable speech-to-text functionalities if needed.
You can install these libraries by running the following command in your virtual environment:

pip install flask transformers torch pydub SpeechRecognition
Note that some additional dependencies might be required depending on your system setup. For these cases, ensure you have Python version >=3.8 installed.


Setup of LM Studio for LLM Usage
Download and Install LM Studio: LM Studio is required to run the llama-3-8b-gpt-4o-GGUF model locally.

Open LM Studio: Launch LM Studio from your desktop or application folder.

Download and Load the Model: From the list of available models in LM Studio, find and download mradermacher/llama-3-8b-gpt-4o-GGUF.

Enable Local API Server:

Inside LM Studio, there is an option to enable the Local API Server.
Ensure the "Enable API" option is activated.
Click the button labeled "Start API Server".

Set the API Port:
Ensure that the API is set to run on port 8080, or specify the port in your application settings to match the API configuration.



default template: IMSyPP
if you want to test the other models:
1 remove the comments related to the model you want to test.
2 comment out the default code

these changes affect two files:
1 app.py
2 result.html
