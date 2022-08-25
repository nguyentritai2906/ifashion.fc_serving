import psycopg2


def query(pids, remote_cids, K):
    # Connect to the PostgreSQL server
    global conn
    global cur
    conn = psycopg2.connect(database="ifashion",
                            user="ifashion",
                            host="db",
                            port="5432")
    # Create a cursor
    cur = conn.cursor()
    candidates = []
    for pid in pids:
        for remote_cid in remote_cids:
            # Find parent category id of question
            cid_sql = """SELECT cid FROM product_category WHERE pid=%s;"""
            cid = fetchone(cid_sql, (pid, ))[0]
            parent_id_sql = """SELECT parent_id FROM category WHERE id=%s;"""
            question_parent_cid = fetchone(parent_id_sql, (cid, ))[0]

            # Find parent category id of answer and its index in table category
            parent_id_sql = """SELECT id, parent_id FROM category WHERE remote_id=%s;"""
            answer_category_index, answer_parent_cid = fetchone(
                parent_id_sql, (remote_cid, ))

            # Find index of typespace embedding in embedding matrix
            # Where cid_1 is question_parent_cid and cid_2 is answer_parent_cid
            index_typespace_sql = """SELECT index FROM typespace
                        WHERE (cid_1=%s AND cid_2=%s)
                            OR (cid_1=%s AND cid_2=%s);"""
            index_typespace = fetchone(index_typespace_sql, [
                question_parent_cid, answer_parent_cid, answer_parent_cid,
                question_parent_cid
            ])[0]

            # Find all images id for this question product
            iids_sql = """SELECT id FROM image WHERE pid=%s;"""
            quenstion_iids = tuple(i[0] for i in fetchall(iids_sql, [pid]))

            # Find all product id has the same category with answer category
            pid_sql = """SELECT pid FROM product_category WHERE cid=%s;"""
            answer_pid_list = tuple(
                i[0] for i in fetchall(pid_sql, [answer_category_index]))
            # Find all image id for these products
            iid_sql = """SELECT id FROM image WHERE pid IN %s;"""
            answer_iid_list = tuple(
                i[0] for i in fetchall(iid_sql, [answer_pid_list]))

            # For each image id in question product iid
            for qiid in quenstion_iids:
                # Find embedding of this image
                qemb_sql = """SELECT emb FROM outfit_recom_emb_%s WHERE iid=%s;"""
                qemb = fetchone(qemb_sql, [index_typespace, qiid])[0]

                aemb_sql = """SELECT iid, cube_distance(emb, %s) FROM outfit_recom_emb_%s
                WHERE iid IN %s ORDER BY emb <-> %s LIMIT %s;"""
                candidate = fetchall(
                    aemb_sql,
                    [qemb, index_typespace, answer_iid_list, qemb, K])
                candidates.extend(candidate)

    answers = sorted(candidates, key=lambda tup: tup[1])[:K]
    answers_iids = tuple(i[0] for i in answers)
    res_pid_sql = """SELECT pid FROM image WHERE id IN %s;"""
    results = [i[0] for i in fetchall(res_pid_sql, [answers_iids])]

    cur.close()
    return results


def fetchone(sql, args):
    cur.execute(sql, args)
    result = cur.fetchone()
    conn.commit()
    return result


def fetchall(sql, args):
    cur.execute(sql, args)
    result = cur.fetchall()
    conn.commit()
    return result

