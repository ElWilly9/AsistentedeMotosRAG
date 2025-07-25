import speech_recognition as sr
import pyttsx3 #Text-to-Speech
import re

def query_voz():
    r = sr.Recognizer()

    with sr.Microphone() as source:
        print("Habla ahora. La grabación terminará cuando te quedes en silencio...")
        # Ajusta el umbral de energía para detectar silencio automáticamente
        r.adjust_for_ambient_noise(source, duration=0.8)
        audio = r.listen(source, timeout=None, phrase_time_limit=None)
        # Guardar el audio en un archivo WAV
        with open("audio/query_user.wav", "wb") as f:
            f.write(audio.get_wav_data())
        try:    
            print("Espere un momento, procesando...")
            text = r.recognize_google(audio, language='es-ES')
            print("Texto reconocido:", text)
            engine = pyttsx3.init()
            engine.setProperty('rate', 200)
            engine.setProperty('volume', 1)
            # Seleccionar voz en español
            for voz in engine.getProperty('voices'):
                if 'spanish' in voz.name.lower() or 'es' in voz.id.lower():
                    engine.setProperty('voice', voz.id)
                    break
            engine.say(text)
            engine.save_to_file(text, "audio/query_ia_voz.mp3")  # Guardar el texto a voz en un archivo MP3
            engine.runAndWait()
            return text
        except:
            print("Lo siento, no pude entender el audio.")

def limpiar_texto(texto):
    # Elimina asteriscos y caracteres especiales, pero deja letras, números, espacios y puntuación básica
    return re.sub(r"[^a-zA-ZáéíóúÁÉÍÓÚñÑ0-9.,;:¿?!¡()\\-\\s]", "", texto)

def decir_respuesta(texto):
    engine = pyttsx3.init()
    engine.setProperty('rate', 200)
    engine.setProperty('volume', 1)
    # Seleccionar voz en español
    for voz in engine.getProperty('voices'):
        if 'spanish' in voz.name.lower() or 'es' in voz.id.lower():
            engine.setProperty('voice', voz.id)
            break
    texto_limpio = limpiar_texto(texto)
    engine.say(texto_limpio)
    engine.save_to_file(texto_limpio, "audio/answer_ia_voz.mp3")
    engine.runAndWait()
