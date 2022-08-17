from PIL import Image
import pandas as pd
import json
import os
import numpy as np


def get_typespace(anchor, pair, typespaces):
    """ Returns the index of the type specific embedding
        for the pair of item types provided as input
    """
    query = (anchor, pair)
    if str(query) not in typespaces:
        query = (pair, anchor)

    return typespaces[str(query)]

def load_embedding_for_typespace(root_path, q_type, cond_type, typespace):
    fname = f'embeddings_{q_type}_{cond_type}_{typespace}.npy'
    if not os.path.exists(os.path.join(root_path, fname)):
        fname = f'embeddings_{cond_type}_{q_type}_{typespace}.npy'
    fpath = os.path.join(root_path, fname)
    typespace_embeddings = np.load(fpath)
    return typespace_embeddings


def find_k_nearest_neighbors(metadata, typespace_embeddings, cond_type, anchor_index, k=5):
    # Get index of candidate items for each question
    candidates = metadata[metadata['type'] == cond_type]
    candidate_indexes = candidates.index

    # Get embeddings for anchor
    anchor_embedding = typespace_embeddings[anchor_index]
    candidate_embeddings = typespace_embeddings[candidate_indexes]

    # Calculate Euclidean distance between anchor and candidate items
    distances = np.linalg.norm(candidate_embeddings - anchor_embedding, axis=1)
    top_k_min_distance_indexes = np.argsort(distances)[:k]
    real_index_in_typespace = candidate_indexes[top_k_min_distance_indexes]
    return real_index_in_typespace


def images_to_sprite(images, size):
    one_square_size = int(np.ceil(np.sqrt(len(images))))
    master_width = size * one_square_size
    master_height = size * one_square_size
    spriteimage = Image.new(
        mode='RGBA',
        size=(master_width, master_height),
        color=(0, 0, 0, 0)
    )

    for count, image in enumerate(images):
        div, mod = divmod(count, one_square_size)
        h_loc = size * div
        w_loc = size * mod
        image = image.resize((size, size))
        spriteimage.paste(image, (w_loc, h_loc))
    return spriteimage.convert('RGB')


def save_sprite(output_path, metadata, image_indexes):
    image_paths = metadata.iloc[image_indexes]['path']
    print(image_paths)
    print('\n'+'-'*10)
    images = [Image.open(path) for path in image_paths]

    sprite = images_to_sprite(images, 112)
    sprite.save(output_path)


def main(K, datadir, outputdir, questions, types, is_save):
    embeddings_metadata = pd.read_csv(f'./{datadir}/embeddings_metadata.tsv', sep='\t')
    typespaces = json.load(open(f'./{datadir}/typespaces.json', 'r'))
    output_path = outputdir
    # data_path = f'./{datadir}/images/'
    embedding_path = f'./{datadir}/embeddings/'

    # Get types and indexes of questions
    questions = questions
    question_rows = [embeddings_metadata[embeddings_metadata['id'] == qid]['type'] 
                                                                  for qid in questions]
    question_types = [qrow.values[0] for qrow in question_rows]
    question_indexes = [qrow.index[0] for qrow in question_rows]
  
    condition_types = types
    # For each question, find the nearest neighbors in the typespace
    # has the same type as the condition type
    candidate_indexes = []
    for q_type, q_index in zip(question_types, question_indexes):
      for condition_type in condition_types:
        typespace = get_typespace(q_type, condition_type, typespaces)
        typespace_embeddings = load_embedding_for_typespace(embedding_path,
                                                            q_type,
                                                            condition_type,
                                                            typespace)

        closest_indexes = find_k_nearest_neighbors(embeddings_metadata,
                                                    typespace_embeddings,
                                                    condition_type,
                                                    q_index,
                                                    k=K*2)
        candidate_indexes.extend(closest_indexes.values)

    # For earch candidate calculate the distance to the question images
    distances = np.zeros((len(candidate_indexes), ), dtype=np.float32)
    for q_type, q_index in zip(question_types, question_indexes):
      for condition_type in condition_types:
        typespace = get_typespace(q_type, condition_type, typespaces)
        typespace_embeddings = load_embedding_for_typespace(embedding_path,
                                                            q_type,
                                                            condition_type,
                                                            typespace)
        anchor_embedding = typespace_embeddings[q_index]
        candidate_embeddings = typespace_embeddings[candidate_indexes]
        distances += np.linalg.norm(candidate_embeddings - anchor_embedding, axis=1)

    # Take the top K closest candidates
    distances = np.array(distances)
    min_distances_indexes = np.argsort(distances)[:K]
    candidate_indexes = list(set(candidate_indexes[i] for i in
                                  min_distances_indexes if i not in
                                  question_indexes))
    # Save the nearest neighbors to a sprite
    if is_save:
        q_output_path = os.path.join(output_path, f'{q_index}')
        os.makedirs(q_output_path, exist_ok=True)
        save_sprite(os.path.join(q_output_path, 'i_sprite.jpg'), embeddings_metadata, question_indexes)

    if is_save:
        c_output_path = os.path.join(output_path, f'{q_index}')
        os.makedirs(c_output_path, exist_ok=True)
        save_sprite(os.path.join(c_output_path, 'o_sprite.jpg'), embeddings_metadata, candidate_indexes)

    return question_indexes, candidate_indexes
