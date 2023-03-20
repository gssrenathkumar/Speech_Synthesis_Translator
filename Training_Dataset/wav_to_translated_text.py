from pydub import AudioSegment
import speech_recognition as sr
from pydub.silence import split_on_silence
from translate import Translator

# create a speech recognition object
reg = sr.Recognizer()


transcript = []

# a function that splits the audio file into chunks and applies speech recognition
def audio_conversion_to_text(path):
    """
    Splitting the large audio file into parts
    and apply speech recognition on each of these parts
    """
    print("Data in english format:")
    # open the audio file using pydub
    sound = AudioSegment.from_wav(path)
    # split audio sound where silence is 200 miliseconds or more and get parts
    parts = split_on_silence(sound,
                              min_silence_len=100,
                              silence_thresh=sound.dBFS-14,
                              keep_silence=500,
                              )
    folder_name = "wavs"
    # create a directory to store the audio parts
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # Processing each and every parts
    for i, audio_parts in enumerate(parts, start=1):
        # export audio chunk and save it in
        # the `folder_name` directory.
        chunk_filename = os.path.join(folder_name, f"{i}.wav")
        audio_parts.export(chunk_filename, format="wav")
        # recognize the parts
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = reg.record(source)
            # try converting it to text
            try:
                text = reg.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                pass
            else:
                text = f"{text.capitalize()}. "
                with open('textfile.txt', 'w') as f:
                    for _ in transcript:
                        f.write(whole_text)
                transcript.append(text)
                print(text)
                whole_text += text + '\n'
    # return the text for all parts detected
    return whole_text


audio_conversion_to_text(
    'krish_naik.wav')

translated_text = []

language = input("Enter the language to translate to: ")

if language == 'fr' or language=="french":
    print("/n")
    print("/n")
    print("Data in French format:")
    translate_text_french = Translator(from_lang='en', to_lang='fr')
    for i in transcript:
        translated_text.append(translate_text_french.translate(i))

    with open('translated_textfile.txt', 'w') as f:
        for i in translated_text:
            print(i)
            f.write(i + '\n')
else:
    pass