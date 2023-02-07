'''
Module for Fastapi endpoints.
'''
import asyncio
from fastapi import FastAPI, File
from fastapi.responses import StreamingResponse

from .utils import array_to_bytes
from .model import nst_generator

app = FastAPI()

DELIMITER = b'--DELIMITER--'


@app.post("/nst")
async def neural_style_trasnfer(image_file: bytes = File(...),
                                style_file: bytes = File(...)):
    '''
    Neural Style Transfer images stream.
    '''
    return StreamingResponse(streamer(nst_generator(image_file, style_file)))


async def streamer(gen):
    '''
    Wrapper for generator for catching closed connections.
    '''
    try:
        for img in gen:
            img_bytes = array_to_bytes(img)
            yield img_bytes
            yield DELIMITER
    except asyncio.CancelledError:
        print("Caught cancelled error")
