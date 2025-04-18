#!/usr/bin/env python
# coding: utf-8

# In[2]:


#pip install fastapi uvicorn pydub speechrecognition


# In[ ]:


from fastapi import FastAPI, File, UploadFile, HTTPException
import librosa
import numpy as np
import uvicorn
import nest_asyncio
import io

nest_asyncio.apply()

app = FastAPI()

@app.post("/process-audio/")
async def process_audio(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        audio_buffer = io.BytesIO(contents)
        
        y, sr = librosa.load(audio_buffer, sr=None)
        
        pauses = np.sum(librosa.effects.split(y, top_db=30))
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_variability = np.std(pitches)
        speech_rate = len(y) / sr
        
        return {
            "pauses": pauses,
            "pitch_variability": pitch_variability,
            "speech_rate": speech_rate
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)


# In[ ]:




