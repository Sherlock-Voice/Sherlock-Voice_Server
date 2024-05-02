# main.py
import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from google.cloud import speech
import io
import wave
from app.textrank_model import *
import numpy as np

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/user1/sherlockvoice_server/app/keyofgstt.json"

app = FastAPI()

# 결과를 저장할 변수
result = {}

def get_sample_rate(file):
    with wave.open(file, "rb") as wave_file:
        return wave_file.getframerate()
    
def get_sample_channels(file):
    with wave.open(file, "rb") as wave_file:
        return wave_file.getnchannels()

async def transcribe_audio(file_content: bytes, filename: str):
    client = speech.SpeechClient()
    
    # 합성음성 판단
    # is_synthetic = model1.predict(content)
    
    sample_rate_hertz = get_sample_rate(io.BytesIO(file_content))
    sample_channels = get_sample_channels(io.BytesIO(file_content))
    
    audio = speech.RecognitionAudio(content=file_content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hertz,
        language_code="ko-KR",
        model="default",
        audio_channel_count=sample_channels,
        enable_automatic_punctuation=True,
        enable_word_confidence=True,
        enable_word_time_offsets=True,
    )

    operation = client.long_running_recognize(config=config, audio=audio)

    print("Waiting for operation to complete...")
    response = operation.result(timeout=90)
    
    transcriptions = {}
    if response.results:
        for res in response.results: 
            print("Transcript: {}".format(res.alternatives[0].transcript))
            
            # 결과를 임시 딕셔너리에 저장
            transcriptions[filename] = res.alternatives[0].transcript

            # 키워드 및 키워드 문장 출력
            print("Keywords: {}".format(summarize_keywords(transcriptions)))

    else:
        print("Not able to transcribe the audio file")
    
    # 결과를 전역 변수에 업데이트
    result[filename] = transcriptions[filename]

@app.post("/upload/")
async def create_upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # Truncate filename to 200 characters
    filename = file.filename[:200]

    # Rest of the code...
    file_content = await file.read()
    background_tasks.add_task(transcribe_audio, file_content, filename)
    return {"filename": filename}


@app.get("/waiting/{filename}")
async def waiting(filename: str):
    # 결과 준비 확인
    if filename in result:
        return {"status": "ready"}
    else:
        return {"status": "processing"}

@app.get("/result/{filename}")
async def get_result(filename: str):
    # 결과 반환
    if filename in result:
        return {"result": result[filename]}
    else:
        return {"error": "No result available"}

