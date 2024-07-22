import numpy as N
from sklearn.metrics.pairwise import cosine_similarity
import pandas as P
from nrcan_p2.data_processing import preprocessing_dfcol
from nrcan_p2.data_processing import preprocessing_str
from nrcan_p2.data_processing import preprocessing_df_filter

from .utils import timer


def convert_text_to_vector_hf(data, model, batch_size=64):
    vectors = []
    for i in range(0, len(data), batch_size):
        vectors.append(model.encode(data[i:i + batch_size]))
    vectors = N.concatenate(vectors, axis=0)
    return vectors


def rank_polygon(descriptive_model, embed_model, data_, desc_col='full_desc', norm=True):
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
    return data_, cos_sim


@timer("Preprocess text")
def preprocess_text(data_df, cols, desc_col='full_desc', filtering=False):
    # ind_invalid = ~sgmc_subset['geometry'].is_valid
    # sgmc_subset.loc[ind_invalid, 'geometry'] = sgmc_subset.loc[ind_invalid, 'geometry'].buffer(0)
    data_ = data_df.copy()

    data_[desc_col] = data_[cols].stack().groupby(level=0).agg(' '.join)
    data_[desc_col] = data_[desc_col].apply(lambda x: x.replace('-', ' - '))

    if filtering:
        pipeline = [
            dfcol_sep_hyphen,
            preprocessing_dfcol.rm_dbl_space,
            preprocessing_dfcol.rm_cid,
            preprocessing_dfcol.convert_to_ascii,
            preprocessing_dfcol.rm_nonprintable,
            preprocessing_df_filter.filter_no_letter,
            preprocessing_dfcol.rm_newline_hyphenation,
            preprocessing_dfcol.rm_newline,
            preprocessing_df_filter.filter_no_real_words_g3letter,
            # preprocessing_df_filter.filter_l80_real_words,
            # preprocessing_dfcol.tokenize_spacy_lg,
            # preprocessing_dfcol.rm_stopwords_spacy,
        ]

        #
        for i, pipe_step in enumerate(pipeline):
            if pipe_step.__module__.split('.')[-1] == 'preprocessing_df_filter':
                data_ = pipe_step(data_, desc_col)
            else:
                data_[desc_col] = pipe_step(data_[desc_col])
            print(f'step {i}/{len(pipeline)} finished')

        #
        post_processing = [
            preprocessing_str.rm_punct,
            preprocessing_str.lower,
            preprocessing_str.rm_newline
        ]

        #
        for i, pipe_step in enumerate(post_processing):
            data_[desc_col] = data_[desc_col].apply(pipe_step)
            print(f'step {i}/{len(post_processing)} finished')

    #
    data_ = data_.drop(columns=['letter_count', 'is_enchant_word', 'word_char_num', 'is_enchant_word_and_g3l',
                                'any_enchant_word_and_g3l', 'real_words', 'real_words_n', 'real_words_perc', 'n_words',
                                'Shape_Area'], errors='ignore')
    data_ = data_.reset_index(drop=True)
    return data_


def dfcol_sep_hyphen(dfcol):
    return dfcol.str.replace('-', ' - ')


def normalize(array):
    return (array - array.min()) * (1 / (array.max() - array.min() + 1e-12))
