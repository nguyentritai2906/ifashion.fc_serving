import psycopg2
from fashion_db import *


def preprocessing(questions, ans_type, K):
    # Connect to the PostgreSQL server
    conn = psycopg2.connect(database="db_fashion", user="db_fashion", password="0", host="localhost", port="5432")
    # Create a cursor
    cur = conn.cursor() 
    candidate_iid = []
    for qid in questions:
        for a_type in ans_type:
            q_cid = find_cid_question(cur, conn, qid)
            a_cid = find_cid_answer(cur, conn, a_type)

            index_typespace = find_index_typespace(cur, conn, q_cid, a_cid)
            ids = get_id(qid)

            for id in ids:
                typespace_embeddings = get_embedding(cur, conn, id, index_typespace)
                ans_iid = find_answer_embedding(cur, conn, typespace_embeddings, index_typespace, K)
                candidate_iid.append(cur, conn, ans_iid)
    cur.close()

    return candidate_iid
