
def get_id(cur, conn, pid):
    sql = """SELECT id FROM image WHERE pid=%s;"""
    cur.execute(sql, [pid])
    id = cur.fetchall()
    conn.commit()
    return [i[0] for i in id]


def find_cid_question(cur, conn, qid):
    sql = """SELECT cid FROM product_category WHERE pid=%s;"""
    cur.execute(sql, [qid])
    cid = cur.fetchone()[0]
    conn.commit()
    return cid


def find_cid_answer(cur, conn, aid):
    sql = """SELECT parent_id FROM category WHERE remote_id=%s;"""
    cur.execute(sql, [aid])
    cid = cur.fetchone()[0]
    conn.commit()
    return cid


def find_index_typespace(cur, conn, qcid, acid):
    sql = """SELECT index FROM typespace 
                WHERE (cid_1=%s AND cid_2=%s)
                    OR (cid_1=%s AND cid_2=%s);"""
    cur.execute(sql, [qcid, acid, acid, qcid])
    index = cur.fetchone()[0]
    conn.commit()
    return index


def get_embedding(cur, conn, qid, typespace):
    sql = """SELECT outfit_recom_emb FROM outfit_recom_emb_%s WHERE iid=%s;"""
    cur.execute(sql, [typespace, qid])
    outfit_recom_emb = cur.fetchone()[0]
    conn.commit()
    return outfit_recom_emb


def find_answer_embedding(cur, conn, input_embedding, index_typespace, k):
    # Insert clothes embedding
    sql = """SELECT iid, outfit_recom_emb, cube_distance(outfit_recom_emb, %s) FROM outfit_recom_emb_%s ORDER BY outfit_recom_emb <-> %s LIMIT %s;"""
    # Excute sql
    cur.execute(sql, (input_embedding, index_typespace, input_embedding, k))  
    result = cur.fetchall() 
    conn.commit() 
    return result


def get_pid(cur, conn, iid):
    sql = """SELECT pid FROM image WHERE id=%s"""
    cur.execute(sql, [iid])
    pid = cur.fetchone[0]
    conn.commit()
    return pid