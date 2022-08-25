import ast
import asyncio
import logging

import cv2
import numpy as np
import psycopg2
import requests
from faust import App

from grpc_recommend_api import grpc_infer, preprocess_image
from query import query

app = App('outfit_app', broker='kafka', value_serializer='raw')
logger = logging.getLogger('outfit_app')

BATCH_SIZE = 8
TIMEOUT = 0.1

insert_pipe_encoder = app.topic('insert_pipe_encoder')
insert_pipe_insert = app.topic('insert_pipe_insert')
outfit_query_in = app.topic('outfit_query_in')
outfit_query_out = app.topic('outfit_query_out')

# Handler for inserting to database
conn = psycopg2.connect(database="ifashion",
                        user="ifashion",
                        host="db",
                        port="5432")
conn.autocommit = True


@app.agent(insert_pipe_encoder)
async def insert_pipe_encoder(stream):
    async for record in stream.take(BATCH_SIZE, within=TIMEOUT):
        iids_embs = []
        for value in record:
            iid, image_link = ast.literal_eval(value.decode())

            response = requests.get(image_link[0])
            input_image = response.content
            input_image = cv2.imdecode(np.frombuffer(input_image, np.uint8),
                                       cv2.IMREAD_UNCHANGED)
            input_image = preprocess_image(input_image)

            output_embedding = grpc_infer(input_image)  # list (4288, )
            output_embedding = np.array(output_embedding).reshape((67, 64))
            iids_embs.append((iid, output_embedding))

        tasks = []
        for iid, emb in iids_embs:
            task = asyncio.create_task(
                insert_pipe_insert.send(value=str((iid,
                                                   emb.tobytes())).encode()))
            tasks.append(task)
        await asyncio.gather(*tasks)


@app.agent(outfit_query_in)
async def query_pipe(stream):
    async for record in stream.take(BATCH_SIZE, within=TIMEOUT):
        for value in record:
            client_id, pids, types, k = ast.literal_eval(value.decode())
            answer_pids = query(pids, types, k)

            tasks = []
            task = asyncio.create_task(
                outfit_query_out.send(key=str(client_id),
                                      value=str(answer_pids).encode()))
            tasks.append(task)
            await asyncio.gather(*tasks)


app.main()
