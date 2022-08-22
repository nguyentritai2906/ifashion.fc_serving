import ast
import asyncio
import logging

import psycopg2
from faust import App

from grpc_recommend_api import grpc_infer
from preprocessing import preprocessing

app = App('outfit_app', broker='kafka://localhost:9093')
logger = logging.getLogger('outfit_app')

BATCH_SIZE = 8
TIMEOUT = 0.1

insert_pipe_encoder = app.topic('insert_pipe_encoder', value_type=bytes)
insert_pipe_insert = app.topic('insert_pipe_insert', value_type=bytes)
query_pipe_in = app.topic('query_pipe_in', value_type=bytes)
query_pipe_out = app.topic('query_pipe_out', value_type=bytes)

# Handler for inserting to database
conn = psycopg2.connect(database="clothes",
                        user="nttai",
                        host="localhost",
                        port="5432")
conn.autocommit = True


@app.agent(insert_pipe_encoder)
async def insert_pipe_encoder(stream):
    async for record in stream.take(BATCH_SIZE, within=TIMEOUT):
        iids_embs = []
        for value in record:
            iid, input_image = ast.literal_eval(value.decode())
            output_embedding = grpc_infer(input_image)  # (67, 64)
            iids_embs.append((iid, output_embedding))

        tasks = []
        for iid, emb in iids_embs:
            task = asyncio.create_task(
                insert_pipe_insert.send(value=str((iid, emb)).encode()))
            tasks.append(task)
        await asyncio.gather(*tasks)


@app.agent(insert_pipe_insert)
async def query_pipe(stream):
    async for record in stream.take(BATCH_SIZE, within=TIMEOUT):
        clien_id, pids, types, k = ast.literal_eval(record.decode())
        answer_pids = preprocessing(pids, types, k)

        tasks = []
        task = asyncio.create_task(
            query_pipe_out.send(key=clien_id, value=str(answer_pids).encode()))
        tasks.append(task)
        await asyncio.gather(*tasks)
