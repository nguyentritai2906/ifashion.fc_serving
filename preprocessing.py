import psycopg2
from fashion_db import *


def preprocessing(questions, ans_type, K):
    # Connect to the PostgreSQL server
    conn = psycopg2.connect(database="q", user="q", password="0", host="localhost", port="5432")
    # Create a cursor
    cur = conn.cursor() 
    candidates = []
    for qid in questions:
        for a_type in ans_type:
            q_cid = find_cid_question(cur, conn, qid)
            a_cid = find_cid_answer(cur, conn, a_type)

            index_typespace = find_index_typespace(cur, conn, q_cid, a_cid)
            ids = get_id(qid)

            for id in ids:
                typespace_embeddings = get_embedding(cur, conn, id, index_typespace)
                candidate = find_answer_embedding(cur, conn, typespace_embeddings, index_typespace, K)
                candidates.extend(candidate)

    answers = sorted(candidates, key=lambda tup: tup[2])[:K]

    cur.close()

    return answers
