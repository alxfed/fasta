# -*- coding: utf-8 -*-
"""https://docs.fast.ai/tutorial.data.html
"""
from fastai.vision import *


def main():
    mnist = untar_data(URLs.MNIST_TINY)
    tfms = get_transforms(do_flip=False)

    '''It's set up with an imagenet structure so we use it to load our training and validation datasets, 
    then label, transform, convert them into ImageDataBunch and finally, normalize them.
    '''
    data = (ImageList.from_folder(mnist)
            .split_by_folder()
            .label_from_folder()
            .transform(tfms, size=32)
            .databunch()
            .normalize(imagenet_stats))

    '''Once your data is properly set up in a DataBunch, we can call data.show_batch() to see what a 
    sample of a batch looks like.
    '''
    # data.show_batch()

    '''Note that the images were automatically de-normalized before being showed with their labels 
    (inferred from the names of the folder). We can specify a number of rows if the default of 5 is too big, 
    and we can also limit the size of the figure.
    '''
    # data.show_batch(rows=3, figsize=(4, 4))

    '''Now let's create a Learner object to train a classifier.
    '''
    learn = cnn_learner(data, models.resnet18, metrics=accuracy)
    learn.fit_one_cycle(1, 1e-2)
    learn.save('mini_train')

    # learn.show_results()
    learn.show_results(ds_type=DatasetType.Train, rows=4, figsize=(10, 10))
    return


if __name__ == '__main__':
    main()
    print('main - done')