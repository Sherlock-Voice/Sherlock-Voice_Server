import os
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from google.cloud import speech
import io
import wave
# from model1 import model_func
# from kobert_model import model_func
# from textrank_model import model_func

app = FastAPI()

result = {}

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/user1/sherlockvoice_server/app/keyofgstt.json"

def transcribe_audio(file: UploadFile):
    client = speech.SpeechClient()
    
    content = file.file.read()
    
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

    print("Waiting for operation to complete...")
    response = operation.result(timeout=90)
    
    if response.results:
        for result in response.results:
            print("Transcript: {}".format(result.alternatives[0].transcript))
            # 결과를 전역 변수에 저장
            result[file.filename] = result.alternatives[0].transcript
                
            # if is_synthetic:
            #     합성음성이면 textrank_model
            #     keywords, summary = textrank_model.predict(text)
            
            # else:
            #     합성음성이 아니면 kobert_model
            #     is_phishing = kobert_model.predict(text)
            
            #     if is_phishing:
            #         보이스피싱이면 textrank_model
            #         keywords, summary = textrank_model.predict(text)
            
            #     else:
            #         return {"message": "This is not a phishing voice."}

            # return {"keywords": keywords, "summary": summary}
    else:
        print("Not able to transcribe the audio file")

def get_sample_rate(file):
    with wave.open(file, "rb") as wave_file:
        return wave_file.getframerate()
    
def get_sample_channaels(file):
    with wave.open(file, "rb") as wave_file:
        return wave_file.getnchannels()

@app.post("/upload/")
async def create_upload_file(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    # 백그라운드 작업으로 오디오 파일 처리 함수를 추가
    background_tasks.add_task(transcribe_audio, file)
    return {"filename": file.filename}

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
