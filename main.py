import os
from fastapi import FastAPI, UploadFile, File
from google.cloud import speech
import io
import wave
# from model1 import model_func
# from kobert_model import model_func
# from textrank_model import model_func

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/user1/sherlockvoice_server/app/keyofgstt.json"

app = FastAPI()

def get_sample_rate(file):
    with wave.open(file, "rb") as wave_file:
        return wave_file.getframerate()
    
def get_sample_channaels(file):
    with wave.open(file, "rb") as wave_file:
        return wave_file.getnchannels()

@app.post("/upload/")
async def create_upload_file(file: UploadFile = File(...)):
    client = speech.SpeechClient()
    
    content = await file.read()
    
    # 합성음성 판단
    # is_synthetic = model1.predict(content)
    
    sample_rate_hertz = get_sample_rate(io.BytesIO(content))
    sample_channaels = get_sample_channaels(io.BytesIO(content))
    
    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hertz,
        language_code="ko-KR",
        model="default",
        audio_channel_count=sample_channaels,
        enable_automatic_punctuation=True,
        enable_word_confidence=True,
        enable_word_time_offsets=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)

# Page of the waiting for the transcribe result
# @app.get("/waiting/")
# async def waiting():
#     return {"message": "Please wait for operation to complete..."}

    print("Waiting for operation to complete...")
    response = operation.result(timeout=90)
    
    if response.results:
        for result in response.results:
            print("Transcript: {}".format(result.alternatives[0].transcript))        
        return {"filename": file.filename}
    else:
        print("Not able to transcribe the audio file")

    # if is_synthetic:
    #     # 합성음성이면 textrank_model
    #     keywords, summary = textrank_model.predict(text)
    # else:
    #     # 합성음성이 아니면 kobert_model
    #     is_phishing = kobert_model.predict(text)
    #     if is_phishing:
    #         # 보이스피싱이면 textrank_model
    #         keywords, summary = textrank_model.predict(text)
    #     else:
    #         return {"message": "This is not a phishing voice."}

    # return {"keywords": keywords, "summary": summary}
