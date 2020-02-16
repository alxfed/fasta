# -*- coding: utf-8 -*-
"""https://docs.fast.ai/tutorial.inference.html#Text
"""
from fastai.text import *


def main():
    imdb = untar_data(URLs.IMDB_SAMPLE)
    data_lm = load_data(imdb)

    data_lm = (TextList.from_csv(imdb, 'texts.csv', cols='text')
               .split_by_rand_pct()
               .label_for_lm()
               .databunch())
    data_lm.save()

    data_lm.show_batch()

    learn = language_model_learner(data_lm, AWD_LSTM)
    learn.fit_one_cycle(2, 1e-2)
    learn.save('mini_train_lm')
    learn.save_encoder('mini_train_encoder')

    print(learn.show_results())

    data_clas = (TextList.from_csv(imdb, 'texts.csv', cols='text', vocab=data_lm.vocab)
                 .split_from_df(col='is_valid')
                 .label_from_df(cols='label')
                 .databunch(bs=42))

    print(data_clas.show_batch())

    learn = text_classifier_learner(data_clas, AWD_LSTM)
    learn.load_encoder('mini_train_encoder')
    learn.fit_one_cycle(2, slice(1e-3, 1e-2))
    learn.save('mini_train_clas')

    print(learn.show_results())
    return


if __name__ == '__main__':
    main()
    print('main - done')