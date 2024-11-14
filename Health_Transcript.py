import os
import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import tkinter as tk
from tkinter import messagebox
import threading

# Initialize the recognizer and translator
recognizer = sr.Recognizer()
translator = Translator()

# Function to recognize speech and translate to English, displaying it only
def recognize_and_translate_speech():
    with sr.Microphone() as source:
        print("Listening...")
        while True:
            try:
                # Adjust for ambient noise and record audio
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source)
                
                # Recognize speech in any language
                text = recognizer.recognize_google(audio)
                print(f"Recognized: {text}")
                
                # Display recognized text live in the input box
                text_input.delete("1.0", tk.END)
                text_input.insert(tk.END, text)
                
                # Translate to English
                translated_text = translator.translate(text, dest='en').text
                print(f"Translated: {translated_text}")
                
                # Update original and translated text displays
                original_text_display.delete("1.0", tk.END)
                original_text_display.insert(tk.END, text)
                
                translated_text_display.delete("1.0", tk.END)
                translated_text_display.insert(tk.END, translated_text)

            except sr.UnknownValueError:
                print("Sorry, I could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")

# Function to save and play translated text as speech when "Speak" button is clicked
def speak_translation():
    translated_text = translated_text_display.get("1.0", tk.END).strip()
    
    if not translated_text:
        messagebox.showwarning("Error", "No translation available.")
        return
    
    # Convert text to speech and play audio only when "Speak" is pressed
    tts = gTTS(translated_text, lang='en')
    tts.save("translated_audio.mp3")
    os.system("start translated_audio.mp3")  # For Windows
    # For macOS: os.system("afplay translated_audio.mp3")
    # For Linux: os.system("mpg321 translated_audio.mp3")

# Set up the GUI window
window = tk.Tk()
window.title("Speech-to-English Translation")

# Text input and buttons for manual translation
text_input_label = tk.Label(window, text="Live recognized text:")
text_input_label.pack(pady=5)
text_input = tk.Text(window, height=5, width=50)
text_input.pack(pady=5)

# Original and translated text displays
original_text_label = tk.Label(window, text="Original Transcript:")
original_text_label.pack(pady=5)
original_text_display = tk.Text(window, height=5, width=50)
original_text_display.pack(pady=5)

translated_text_label = tk.Label(window, text="Translated Transcript (English):")
translated_text_label.pack(pady=5)
translated_text_display = tk.Text(window, height=5, width=50)
translated_text_display.pack(pady=5)

# Speak button for translated text
speak_button = tk.Button(window, text="Speak", command=speak_translation)
speak_button.pack(pady=5)

# Run speech recognition in a separate thread
speech_thread = threading.Thread(target=recognize_and_translate_speech, daemon=True)
speech_thread.start()

# Run the Tkinter event loop
window.mainloop()



# from predibase import Predibase
# os.environ["PREDIBASE_API_TOKEN"] = "pb_GW8Li822WuK_GutDVN5UBg"
# pb = Predibase(api_token="pb_GW8Li822WuK_GutDVN5UBg")
# lorax_client = pb.deployments.client("mistral-7b-instruct-v0-2")
# def generate_summary(text):
#     input_prompt = (
#         "[INST] Correct any misrecognized words in the text coming from speech recognition and enhance transcription accuracy, especially for medical terms but keep the language, style, and format the same. "
#         "Do not add any extra commentary or changes. Here is the text from speech recognition: "
#         f"{text} [/INST]"
#     )
#     lorax_client = pb.deployments.client("mistral-7b-instruct-v0-2")
#     response = lorax_client.generate(input_prompt)
#     summary = response.generated_text
#     return summary