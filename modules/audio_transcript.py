import speech_recognition as sr

def record_and_transcribe():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        try:
            audio = recognizer.listen(source, timeout=3)
            return recognizer.recognize_google(audio)
        except:
            return ""
