# Healthcare Live Voice Translation Web App with Generative AI

This project provides a GUI-based application that allows users to speak into a microphone. The speech is recognized, translated into English, and displayed in the GUI. The user can also play the translated text as speech using a built-in Text-to-Speech (TTS) functionality.

## Features

- **Speech Recognition:** Captures and transcribes speech in real-time using Google's Speech Recognition API.
- **Translation:** Automatically translates the transcribed text to English using the Google Translate API.
- **Text-to-Speech (TTS):** Converts the translated English text into speech using the Google Text-to-Speech (gTTS) API.
- **GUI Interface:** The user-friendly interface displays both the original recognized text and its English translation. The translated text can be played as speech by clicking the "Speak" button.
- **Predibase Model** For Generative Ai (Most of available models are paid thats why i use this)

## Requirements

- Python 3.6+
- `speechrecognition` library for speech recognition
- `googletrans` library for translation
- `gtts` library for text-to-speech conversion
- `tkinter` for the GUI interface (Usually included with Python)
- An internet connection for Google services

### Activating the Virtual Environment

1. **Activate the Virtual Environment:**

   If you have already created a virtual environment in the project folder, activate it by running the following command:

   - **For Windows:**
     ```bash
     venv\Scripts\activate
     ```

   - **For macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

2. **Usage**

    Run the Application:

    After activating the virtual environment and installing the dependencies, run the script Health_Transcript.py to launch the application.

    ```bash
    python Health_Transcript.py
    ```

3. **How it works:**

The application will listen to your microphone input.
It will recognize your speech and display the text in the GUI.
The recognized text will be translated into English and shown on the GUI.
If the translation is available, you can click the "Speak" button to hear the translated text in English.

4. **GUI Layout:**

Live recognized text: Shows the text that is recognized in real-time from your speech.
Original Transcript: Displays the text as it was recognized (including the original language).
Translated Transcript (English): Displays the translated text in English.
Speak Button: Click to hear the translated text using the Text-to-Speech (TTS) functionality.


5. **Notes**

Make sure your microphone is connected and properly configured before running the program.
The SpeechRecognition module uses Google's API to recognize speech, which requires an active internet connection.
The application listens continuously and updates the GUI with new recognized text and translation.
The TTS system uses gTTS to convert text to speech and saves it as an audio file (translated_audio.mp3), which is then played.


6. **Troubleshooting**
Microphone Issues:

If your microphone is not being recognized, ensure it is properly connected and recognized by your operating system.
You can test microphone functionality using other audio-related applications like a voice recorder.
Translation Errors:

Ensure you have a stable internet connection for the Google Translate API to function correctly.
If the translation is not accurate, ensure the original speech was clear and correctly recognized.
