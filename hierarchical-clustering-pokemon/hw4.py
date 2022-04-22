"""
@author: Eric Choi
"""
from scipy.cluster.hierarchy import dendrogram, linkage
import csv
import math
import numpy.linalg as LA
import numpy as np
import matplotlib.pyplot as plt

def load_data(filepath):
    pokemons = []
    with open(filepath, encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dic = dict()
            dic["#"] = row['#']
            dic["Name"] = row['Name']
            dic["Type 1"] = row['Type 1']
            dic["Type 2"] = row['Type 2']
            dic["Total"] = row['Total']
            dic["HP"] = row['HP']
            dic["Attack"] = row['Attack']
            dic["Defense"] = row['Defense']
            dic["Sp. Atk"] = row['Sp. Atk']
            dic["Sp. Def"] = row['Sp. Def']
            dic["Speed"] = row['Speed']
            dic["Generation"] = row['Generation']
            dic["Legendary"] = row['Legendary']
            pokemons.append(dic)
    return pokemons


def calc_features(row):    
    return np.array([int(row["Attack"]), int(row["Sp. Atk"]), int(row["Speed"]), int(row["Defense"]), int(row["Sp. Def"]), int(row["HP"])], dtype='int64')


def get_min_distance(min, tmp):
    if min[2] >= tmp[2]:
        if min[2] == tmp[2]:
            # equal first index ith
            if min[0] >= tmp[0]:
                if min[0] == tmp[0]:
                    # equal second index jth
                    if min[1] >= tmp[1]:
                        # share same pokemon stat
                        min = tmp
                else:
                    min = tmp
        else:
            min = tmp
    return min


def merge_distance(cluster_dict):
    length = len(cluster_dict)
    min = [np.inf, np.inf, np.inf]  # default to inf for replacement
    # iterate through cluster list (i.e., [1 2 3] --> (1, 2), (1, 3), (2, 3))
    for i in range(length):
        if str(type(cluster_dict[i])).find('int') != -1:
            continue    # already clustered
        for j in range(1, length - i):
            if str(type(cluster_dict[i + j])).find('int') != -1:
                continue    # already clusterd
            tfst = 'tuple' in str(type(cluster_dict[i]))
            tsnd = 'tuple' in str(type(cluster_dict[i+j]))
            # check if multiple pokemons in cluster or not
            if tfst and tsnd:
                # multiple pokemons in both
                distances = max_dist = []
                cluster_list1 = list(cluster_dict[i])
                cluster_list2 = list(cluster_dict[i + j])
                # complete-linkage
                for c1 in cluster_list1:
                    for c2 in cluster_list2:
                        tmp = [i, i + j, LA.norm(c1 - c2)]
                        distances.append(tmp)
                        dist = np.array(distances).T[2]  # get distances only
                        # get index of max distance
                        max_idx = np.argmax(dist)
                        max_dist = distances[max_idx]
                min = get_min_distance(min, max_dist)
            elif tfst and not tsnd:
                # multiple pokemons in first cluster
                distances = max_dist = []
                cluster_list = list(cluster_dict[i])
                # complete-linkage
                for c in cluster_list:
                    tmp = [i, i + j, LA.norm(c - cluster_dict[i + j])]
                    distances.append(tmp)
                    dist = np.array(distances).T[2]  # get distances only
                    max_idx = np.argmax(dist)       # get index of max distance
                    max_dist = distances[max_idx]
                min = get_min_distance(min, max_dist)
            elif not tfst and tsnd:
                # multiple pokemons in second cluster
                distances = max_dist = []
                cluster_list = list(cluster_dict[i + j])
                # complete-linkage
                for c in cluster_list:
                    tmp = [i, i + j, LA.norm(cluster_dict[i] - c)]
                    distances.append(tmp)
                    dist = np.array(distances).T[2]  # get distances only
                    max_idx = np.argmax(dist)       # get index of max distance
                    max_dist = distances[max_idx]
                min = get_min_distance(min, max_dist)
            else:
                # single pokemon in both
                tmp = [
                    i, i + j, LA.norm(cluster_dict[i] - cluster_dict[i + j])]
                min = get_min_distance(min, tmp)
    return min


def hac(features):    
    #print(linkage(features, method='complete'))
    flen = len(features)
    # (n-1) x 4 array
    res = np.zeros((flen-1, 4))
    # track clusters
    cluster_dict = dict()
    for i in range(flen):
        cluster_dict[i] = features[i]

    # compute complete-linkage
    count = flen
    for r in range(flen - 1):
        # get minimum distance indices
        indices = merge_distance(cluster_dict)
        # clusters to be merged
        c1 = cluster_dict[indices[0]]
        c2 = cluster_dict[indices[1]]
        # check if multiple pokemons in cluster or not
        # if tuple, then multiple pokemons, otherwise single pokemon
        tfst = str(type(c1)).find('tuple')
        tsnd = str(type(c2)).find('tuple')
        l1 = l2 = []
        if tfst == -1 and tsnd == -1:
            # single pokemon
            l1 = [c1]
            l2 = [c2]
        else:
            # multiple pokemons
            if tfst != -1:
                l1 = list(c1)
            else:
                l1 = [c1]
            if tsnd != -1:
                l2 = list(c2)
            else:
                l2 = [c2]
        # merge clusters
        ncluster = tuple(np.append(l1, l2, axis=0))
        fst_idx = indices[0]
        snd_idx = indices[1]
        # update output
        res[r][0] = fst_idx         # index of first cluster
        res[r][1] = snd_idx         # index of seoncd cluster
        res[r][2] = indices[2]      # distance
        res[r][3] = len(ncluster)   # num of elements in cluster
        # add into cluster list
        cluster_dict[count] = ncluster
        count += 1
        # remove from cluster list
        cluster_dict[fst_idx] = -1
        cluster_dict[snd_idx] = -1
    
    return res


def imshow_hac(Z):
    plt.figure()
    dn = dendrogram(Z)
    plt.show()


if __name__ == "__main__":
    #n = 10
    for n in range(2, 21):
        #Z = linkage([calc_features(feature)
                    #for feature in load_data('Pokemon.csv')][:n], 'complete')
        #plt.figure()
        #dn = dendrogram(Z)
        #plt.show()
        Z = hac([calc_features(feature)
                for feature in load_data('Pokemon.csv')][:n])
        imshow_hac(Z)
