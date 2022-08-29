import psycopg2

CAT_CONSTRAINT = {
    12: [19, 18, 21, 22, 24, 26, 29, 31, 30],
    13: [19, 18, 21, 22, 24, 26, 29, 31, 30],
    14: [19, 18, 21, 22, 24, 26, 29, 31, 30],
    15: [19, 18, 21, 22, 24, 26, 29, 31, 30],
    16: [19, 18, 21, 22, 24, 26, 29, 31, 30],
    17: [24, 29, 26, 25, 31, 12, 13, 14, 15, 16],
    18: [24, 29, 26, 25, 31, 12, 13, 14, 15, 16],
    19: [24, 26, 23, 31],
    20: [26, 23, 33],
    21: [23, 26, 12, 13, 14, 15, 16],
    22: [24, 29, 31, 30, 12, 13, 14, 15, 16],
    23: [20, 19, 21, 33],
    24: [19, 18, 21, 22, 12, 13, 14, 15, 16],
    25: [19, 18, 21, 22, 12, 13, 14, 15, 16],
    26: [20, 19, 18, 33],
    27: [20, 19, 21, 18, 22, 12, 13, 14, 15, 16],
    28: [20, 19, 21, 18, 22, 12, 13, 14, 15, 16],
    29: [20, 19, 21, 18, 22, 12, 13, 14, 15, 16],
    30: [20, 19, 21, 18, 12, 13, 14, 15, 16],
    31: [19, 21, 18, 22, 12, 13, 14, 15, 16],
    33: [20, 19, 18, 23],
}


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


def generate_outfit(pid, k):
    # Connect to the PostgreSQL server
    global conn
    global cur
    conn = psycopg2.connect(database="ifashion",
                            user="ifashion",
                            host="db",
                            port="5432")
    # Create a cursor
    cur = conn.cursor()

    # Product embedding
    emb_sql = """SELECT emb FROM image_search_emb INNER JOIN image ON image_search_emb.iid = image.id
                  WHERE pid=%s"""
    try:
        emb = fetchone(emb_sql, (pid, ))[0]
    except TypeError:
        return [None]

    # Category of product
    cid_sql = """SELECT cid FROM product_category WHERE pid=%s;"""
    cid = [i[0] for i in fetchall(cid_sql, (pid, )) if i not in range(11)]
    cid = cid[0]
    # Parent category of product
    parent_id_sql = """SELECT parent_id FROM category WHERE id=%s;"""
    question_parent_cid = fetchone(parent_id_sql, (cid, ))[0]

    # Same category products
    sql = """Select pid from product_category where cid = %s;"""
    search_pids = tuple(i[0] for i in fetchall(sql, [question_parent_cid]))
    if question_parent_cid == 10 and cid != 33:
        vest_sql = """Select pid from product_category where cid = 33;"""
        vest_pids = tuple(i[0] for i in fetchall(vest_sql, []))
        search_pids = tuple(x for x in search_pids if x not in vest_pids)

    # Find similar product ids to replace input pid
    sql = """SELECT pid FROM image_search_emb INNER JOIN image ON image_search_emb.iid = image.id
            WHERE image.pid IN %s ORDER BY emb <-> %s LIMIT %s;"""
    candidate_pids = tuple(x[0] for x in fetchall(sql, [search_pids, emb, k]))

    # Parent ids
    #categories = {'top': 5, 'bottom': 2, 'shoes': 9, 'outerwear': 10}
    categories = {'top': 5, 'bottom': 2, 'outerwear': 10}
    parent_cids = tuple(v for v in categories.values()
                        if v != question_parent_cid)

    compatible_categories = tuple(x for x in CAT_CONSTRAINT[cid])
    # Return 2 outfits each has 3 products
    outfits = []
    for pid in candidate_pids:
        # iids for this product
        iids_sql = """SELECT id FROM image WHERE pid=%s;"""
        quenstion_iids = [i[0] for i in fetchall(iids_sql, [pid])]
        answer_pids = [pid]

        for cur_cid in parent_cids:
            # Find index of typespace embedding in embedding matrix
            # Where cid_1 is question_parent_cid and cid_2 is answer_parent_cid
            index_typespace_sql = """SELECT index FROM typespace
                        WHERE (cid_1=%s AND cid_2=%s)
                            OR (cid_1=%s AND cid_2=%s);"""
            index_typespace = fetchone(
                index_typespace_sql,
                [question_parent_cid, cur_cid, cur_cid, question_parent_cid
                 ])[0]

            # same parent category products
            iid_sql = """SELECT image.id, image.pid
                        FROM image INNER JOIN product_category on product_category.pid = image.pid
                        WHERE cid=%s;"""
            iid_pid = tuple(tuple(i) for i in fetchall(iid_sql, [cur_cid]))
            pid_sql = """SELECT image.id, image.pid
                        FROM image INNER JOIN product_category on product_category.pid = image.pid
                        WHERE cid IN %s;"""
            com_pid = tuple(
                tuple(i) for i in fetchall(pid_sql, [compatible_categories]))
            answer_iid_tup = set(iid_pid).intersection(com_pid)
            answer_iid_tup = tuple(x[0] for x in answer_iid_tup)

            # Use previous product choices to recommend last item
            answer_iids = []
            if len(answer_pids) > 1:
                answer_sql = """SELECT id FROM image WHERE pid=%s;"""
                answer_iids = [
                    i[0] for i in fetchall(answer_sql, [answer_pids[-1]])
                ]

            candidates = []
            # For each image id in question product iid
            for qiid in quenstion_iids + answer_iids:
                # Find embedding of this image
                qemb_sql = """SELECT emb FROM outfit_recom_emb_%s WHERE iid=%s;"""
                qemb = fetchone(qemb_sql, [index_typespace, qiid])[0]

                aemb_sql = """SELECT pid, cube_distance(emb, %s)
                FROM image INNER JOIN outfit_recom_emb_%s ON outfit_recom_emb_%s.iid = image.id
                WHERE image.id IN %s ORDER BY emb <-> %s LIMIT %s;"""
                candidate = fetchall(aemb_sql, [
                    qemb, index_typespace, index_typespace, answer_iid_tup,
                    qemb, k
                ])
                candidates.extend(candidate)

            candidates = [x for x in candidates if x[0] not in outfits[-1]
                          ] if len(outfits) > 0 else candidates

            answer_pids.append(
                sorted(candidates, key=lambda tup: tup[1])[0][0])
        outfits.append(answer_pids)

    cur.close()
    print(outfits)
    return outfits


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
