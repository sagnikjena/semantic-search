import pandas as pd
from fse import IndexedList
from fse.models import SIF
from utils.helper import init_model, process_text, get_relevant_words


def semantic_search_sif(search_string, sif_model, data, em_model, model_type, indexed_docs, n_top=10):
    process_search_str = process_text(search_string)
    matching_idx = []
    matched_data = sif_model.sv.similar_by_sentence(sentence=process_search_str,
                                                    model=sif_model, indexable=indexed_docs.items,
                                                    topn=n_top)
    for match in matched_data:
        matching_idx.append((match[1], match[2]))

    result_df = pd.DataFrame(columns=['headlines_matched', 'relevant_words', 'similarity'], index=range(n_top))
    for i in range(n_top):
        result_df['headlines_matched'][i] = data.headline_text[matching_idx[i][0]]
        result_df['relevant_words'][i] = get_relevant_words(search_tok=process_search_str,
                                                            doc_tok=data.processed_headlines[matching_idx[i][0]],
                                                            model=em_model, model_type=model_type)
        result_df['similarity'][i] = matching_idx[i][1]

    return result_df


def main():
    data = pd.read_parquet('../data/sample_data.parquet')
    documents = [doc.split() for doc in data.headline_text]
    docs_idx = IndexedList(documents)

    em_model, mod_ty = init_model(model_type='gensim')

    sif_model = SIF(em_model, workers=6)
    sif_model.train(docs_idx)

    search_headline = input("Search headline: ")
    output = semantic_search_sif(search_string=search_headline, sif_model=sif_model, data=data,
                                 em_model=em_model, model_type=mod_ty, indexed_docs=docs_idx)

    print(output)


if __name__ == '__main__':
    main()





