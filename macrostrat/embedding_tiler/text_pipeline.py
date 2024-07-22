import numpy as N
from sklearn.metrics.pairwise import cosine_similarity
import pandas as P
from .deposit_models import systems_dict

from .utils import timer


def convert_text_to_vector_hf(data, model):
    return model.encode(data)


def rank_polygons_by_deposit_model(model_key: str, embed_model, data_, desc_col='full_desc', norm=True):
    descriptive_model = systems_dict[model_key]
    polygon_vectors = convert_text_to_vector_hf(data_[desc_col].to_list(), embed_model)

    query_vec = {}
    cos_sim = {}

    polygon_vectors_age_range = convert_text_to_vector_hf(data_['age'].to_list(), embed_model)

    for key in descriptive_model:
        query_vec[key] = convert_text_to_vector_hf([descriptive_model[key]], embed_model)
        cos_sim[key] = cosine_similarity(query_vec[key], polygon_vectors)[0]
        if norm:
            cos_sim[key] = normalize(cos_sim[key])

    try:
        cos_sim['age_range'] = cosine_similarity(query_vec['age_range'], polygon_vectors_age_range)[0]
    except Exception as e:
        print("Failed to compute age range score. Skipping ...")
        pass

    bge_all = 0
    for key in cos_sim:
        tmp = cos_sim[key]
        # tmp_color = float_to_color(tmp)
        bge_all += tmp
        data_['bge_' + key] = P.Series(list(tmp))
        # data_['bge_'+key+'_color'] = pd.Series(list(tmp_color))

    bge_all /= len(cos_sim)
    # bge_all_color = float_to_color(bge_all)
    data_['bge_all'] = P.Series(list(bge_all))
    # data_['bge_all_color'] = pd.Series(list(bge_all_color))
    return data_


def rank_polygons(term, embed_model, data, text_col='full_desc'):
    query_vec = convert_text_to_vector_hf([term], embed_model)

    data["similarity"] = N.nan

    g1 = data.groupby(by=text_col)

    group_names = list(g1.groups.keys())

    group_vectors = convert_text_to_vector_hf(group_names, embed_model)
    sim = cosine_similarity(query_vec, group_vectors)[0]
    sim = normalize(sim)

    for group, sim_ in zip(group_names, sim):
        data.loc[g1.get_group(group).index, "similarity"] = sim_

    return data


@timer("Preprocess text")
def preprocess_text(data_df, cols, desc_col='full_desc', filtering=False):
    data_ = data_df

    data_[desc_col] = data_[cols].stack().groupby(level=0).agg(' '.join)
    data_[desc_col] = data_[desc_col].apply(lambda x: x.replace('-', ' - '))
    # Strip extra trailing spaces
    data_[desc_col] = data_[desc_col].str.strip()

    return data_


def normalize(array):
    return (array - array.min()) * (1 / (array.max() - array.min() + 1e-12))
