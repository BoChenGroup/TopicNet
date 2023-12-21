import scipy.io as sio # mat
import numpy as np

topic_list = []
graph_wordnet = sio.loadmat('./dataset/TopicTree_20ng.mat')

for i in range(len(graph_wordnet['graph_topic_name'][0])):
    topic_list.append(len(graph_wordnet['graph_topic_name'][0][i]))
    for j in range(len(graph_wordnet['graph_topic_name'][0][i])):
        print(str(j) + '  ' + graph_wordnet['graph_topic_name'][0][i][j])

# adj = graph_wordnet['graph_topic_adj'][0]
# adj1 = np.transpose(adj[1])
#
# topic1_list = [9, 29, 37, 89, 125, 179, 239, 261, 263, 271]
# for i in topic1_list:
#     for j in range(adj1.shape[1]):
#         if adj1[i, j] > 0:
#            print(str(i) + ':' + str(j))
# print(0)
