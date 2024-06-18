
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def t_sne(args, embeds, labels, epoch, seed, sample_num=2000, show_fig=False):
    """
    visualize embedding by t-SNE algorithm
    :param embeds: embedding of the data
    :param labels: labels
    :param sample_num: the num of samples
    :param show_fig: if show the figure
    :return: figure
    """

    # sampling
    sample_index = np.random.randint(0, embeds.shape[0], sample_num)
    sample_embeds = embeds[sample_index]
    labels = labels[sample_index]

    # t-SNE
    ts = TSNE(n_components=2, init='pca', random_state=0)
    ts_embeds = ts.fit_transform(sample_embeds[:, :])
    # ts_embeds = ts_embeds.cpu()

    # remove outlier
    mean, std = np.mean(ts_embeds, axis=0), np.std(ts_embeds, axis=0)
    for i in range(len(ts_embeds)):
        if (ts_embeds[i] - mean < 3 * std).all():
            np.delete(ts_embeds, i)

    # normalization
    x_min, x_max = np.min(ts_embeds, 0), np.max(ts_embeds, 0)
    norm_ts_embeds = (ts_embeds - x_min) / (x_max - x_min)

    # plot
    # print(max(labels)+1)
    fig = plt.figure()
    # print(labels)
    # print(labels.data[i])
    # f = open('./our_result_xy_{}.csv'.format(cnt), 'a+')
    # print("acm", file=f)
    colorset = ['darkgrey', 'blue', 'red']
    for i in range(norm_ts_embeds.shape[0]):
        # if labels[i]==1 or labels[i] == 2:
        #     continue
        '''
        plt.text(norm_ts_embeds[i, 0], norm_ts_embeds[i, 1], str(int(labels.data[i])),
                 color=plt.cm.Set1(int(labels.data[i]) % args.cluster_num),
                 fontdict={'weight': 'bold', 'size': 7})
        # print(str(norm_ts_embeds[i,0])+","+str(norm_ts_embeds[i, 1]), file=f)
        '''
        '''
        plt.text(norm_ts_embeds[i, 0], norm_ts_embeds[i, 1], str(int(labels[i])),
                 color=plt.cm.Set1(int(labels[i]) % args.cluster_num),
                 fontdict={'weight': 'bold', 'size': 7})
        '''
        plt.text(norm_ts_embeds[i, 0], norm_ts_embeds[i, 1], 'o',
                 color=colorset[int(labels[i])],
                 fontdict={'weight': 'bold', 'size': 7})
    # f.close()


    plt.xticks([])
    plt.yticks([])
    # plt.title('t-SNE', fontsize=14)
    plt.axis('off')
    plt.savefig("{}_{}_{}.png".format(args.dataset, seed, epoch), dpi=1000)
    if show_fig:
        plt.show()
    return fig


def similarity_plot(embedding, label, sample_num=1000, show_fig=True):
    """
    show cosine similarity of embedding or x
    :param embedding: the input embedding
    :param label: the ground truth
    :param sample_num: sample number
    :param show_fig: if show the figure
    :return: the figure
    """
    # sampling
    label_sample = label[:sample_num]
    embedding_sample = embedding[:sample_num, :]

    # sort the embedding based on label
    cat = np.concatenate([embedding_sample, label_sample.reshape(-1, 1)], axis=1)
    arg_sort = np.argsort(label_sample)
    cat = cat[arg_sort]
    embedding_sample = cat[:, :-1]

    # cosine similarity
    norm_embedding_sample = embedding_sample / np.sqrt(np.sum(embedding_sample ** 2, axis=1)).reshape(-1, 1)
    cosine_sim = np.matmul(norm_embedding_sample, norm_embedding_sample.transpose())
    cosine_sim[cosine_sim < 1e-5] = 0

    # figure
    fig = plt.figure()
    sns.heatmap(data=cosine_sim, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.axis("off")

    # plot
    if show_fig:
        plt.show()
    return fig
