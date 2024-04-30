import os
from fastapi import FastAPI, UploadFile, File
from google.cloud import speech
import io
import wave
# from models import model_func

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


'''
# Send the result of transcribe to the AI model
def model_func(file):
    return "This is the result from the model"

@app.post("/final result/")
async def final_result():
    result = model_func(file)
    return {"result": result}
'''