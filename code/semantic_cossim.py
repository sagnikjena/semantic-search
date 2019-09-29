import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from utils.helper import process_text, init_model, get_docs_embedding, text2vec, get_relevant_words


def semantic_search_cossim(search_string, docs_embeddings, em_model, model_type, data, topn=10):
    # clean search text
    process_search_str = process_text(search_string)
    # convert search text to vec
    search_string_vect = np.array(text2vec(process_search_str, em_model)).reshape(1, -1)
    # find cosine similarity b/w search text and document headlines
    cosine_similarities = pd.Series(cosine_similarity(search_string_vect, docs_embeddings)[0])
    # create an empty dataframe to write output to
    result_df = pd.DataFrame(columns=['headlines_matched', 'relevant_words', 'similarity'], index=range(topn))
    k = 0
    # write data to output dataframe by sorting in ascending order of cosine similarity
    for i, j in cosine_similarities.nlargest(int(topn)).iteritems():
        result_df['headlines_matched'][k] = data['headline_text'][i]
        result_df['relevant_words'][k] = get_relevant_words(search_tok=process_search_str,
                                                            doc_tok=data.processed_headlines[i],
                                                            model=em_model, model_type=model_type)
        result_df['similarity'][k] = j
        k += 1

    return result_df


def main():
    # load the pre-trained word embedding model
    pre_trained_model, mod_ty = init_model(model_type='magnitude')
    # read data
    data = pd.read_parquet('../data/sample_data.parquet')
    print(data.shape)
    print(data.head())
    # check if the document embedding file exists
    # if not, then create
    docs_model_file = Path('../model/docs_embedding.parquet')
    if docs_model_file.exists():
        docs_embedding = pd.read_parquet('../model/docs_embeddings.parquet').values
    else:
        docs_embedding = get_docs_embedding(docs_tok=data.processed_headlines, model=pre_trained_model)
    # get search string
    search_headline = input("Search headline: ")
    # get the semantic search output
    output = semantic_search_cossim(search_string=search_headline, docs_embeddings=docs_embedding,
                                    em_model=pre_trained_model, model_type=mod_ty, data=data)
    print(output)


if __name__ == '__main__':
    main()
