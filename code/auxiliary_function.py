import re
import os
import json
import jieba
import numpy as np
import networkx as nx
from torchtext.data import Field


def construct_label_dict(weibo_txt_path, rate):
    """
    Construct dict (eid: label)

    :param weibo_txt_path: path of weibo_txt
    :param rate: rate of data for training
    :return: dict for training, dict for validation
    """

    train_label_dict = {}
    val_label_dict = {}
    pattern = re.compile(":([0-9]*?)\t")
    weibo_txt = open(weibo_txt_path).readlines()
    weibo_txt_len = int(len(weibo_txt) * rate)

    for i in range(len(weibo_txt)):
        rp = pattern.findall(weibo_txt[i])

        if i < weibo_txt_len:
            train_label_dict[rp[0]] = rp[1]
        else:
            val_label_dict[rp[0]] = rp[1]

    return train_label_dict, val_label_dict


def text_collect(weibo_json_path, eid_list):
    """
    Collect text for vocab building

    :param weibo_json_path: path of JSON file
    :param eid_list: list of eid
    :return: list of text
    """

    text_list = []

    for eid in eid_list:
        file = "{}.json".format(eid)
        file_path = os.path.join(weibo_json_path, file)
        with open(file_path, "r", encoding="utf-8") as f:
            f_content = json.loads(f.read())

        for item in f_content:
            node_text = item["text"]
            text_list.append(node_text)

    return text_list


def stop_word(stop_word_path):
    """
    Collect Chinese_stop_word

    :param stop_word_path: path of Chinese_stop_word
    :return: list of Chinese_stop_word
    """

    with open(stop_word_path, "r", encoding="utf-8") as f:
        f_content = f.readlines()

    f_content = list(map(lambda x: x.strip(), f_content))

    return f_content


def word_div(sentence):
    """
    Function for sentence segmentation

    :param sentence: sentence that need to be segmented
    :return: word list of segmented sentence
    """

    stopwords = stop_word("stop_word/Chinese_stop_words.txt")
    pattern = re.compile("[^\u4e00-\u9fa5]")
    sentence = pattern.sub("", sentence)

    return [word.strip() for word in jieba.cut(sentence) if word.strip() not in stopwords]


def vocab_build(text, st_length):
    """
    Build vocab

    :param text: word list of segmented sentences
    :param st_length: fixed length of sentences
    :return: text_field
    """

    Text = Field(sequential=True, fix_length=st_length, tokenize=word_div)
    corpus = [word_div(example) for example in text]
    Text.build_vocab(corpus)

    return Text


def word_transform(word_list, text, st_length):
    """
    Use vocab to transform word list

    :param word_list: word list of segmented sentence
    :param text: text_field
    :param st_length: fixed length of sentences
    :return: transformed word list
    """

    transform_res = [1] * st_length
    idx = 0

    for word in word_list:
        if idx < st_length:
            transform_res[idx] = text.vocab.stoi[word]
            idx += 1

    return np.asarray(transform_res, dtype=np.longlong)


def json_to_network(json_file_path, vocab, st_length):
    """
    Use JSON file to construct network

    :param json_file_path: path of JSON file
    :param vocab: text_field
    :param st_length: fixed length of sentences
    :return: network constructed by JSON file, id of source post
    """

    graph = nx.DiGraph()

    with open(json_file_path, "r", encoding="utf-8") as f:
        f_content = json.loads(f.read())

    eid = f_content[0]["id"]
    edges_list = []
    text_list = []

    for item in f_content:
        # post's feature: count of re_posts/ count of attitudes/ count of comments
        node_reposts_count = item["reposts_count"]
        node_attitudes_count = item["attitudes_count"]
        node_comments_count = item["comments_count"]
        text_list.append(word_transform(word_div(item["text"]), vocab, st_length))

        # poster's feature: count of bi_followers/count of friends/location/
        # count of followers/count of posts/verified or not/count of favourites/gender
        node_bi_followers_count = item["bi_followers_count"]
        node_friends_count = item["friends_count"]
        node_province = item["province"]
        node_followers_count = item["followers_count"]
        node_statuses_count = item["statuses_count"]
        node_verified = 1 if item["verified"] else 0
        node_favourites_count = item["favourites_count"]
        node_gender = 1 if item["gender"] == "m" else 0

        # id of current/parent post which could be used to construct edges
        node_mid = item["mid"]
        node_parent = item["parent"]

        if node_parent:
            edges_list.append((node_parent, node_mid))

        graph.add_node(node_mid, reposts_count=node_reposts_count, attitudes_count=node_attitudes_count,
                       comments_count=node_comments_count, bi_followers_count=node_bi_followers_count,
                       friends_count=node_friends_count, province=int(node_province),
                       followers_count=node_followers_count, statuses_count=node_statuses_count,
                       verified=node_verified, gender=node_gender, favourites_count=node_favourites_count)

    graph.add_edges_from(edges_list)

    return graph, eid, text_list


def get_node_info(graph):
    """
    Extract feature matrix of nodes from network

    :param graph: network
    :return: feature matrix of nodes
    """

    node_info_list = []
    node_full_info = list(graph.nodes(data=True))

    for item in node_full_info:
        node_info_list_ind = []
        node_info = item[1]
        for key, value in node_info.items():
            node_info_list_ind.append(value)
        node_info_list.append(node_info_list_ind)

    return node_info_list


def feature_normalize(data):
    """
    Normalize feature matrix

    :param data: feature matrix
    :return: normalized feature matrix
    """

    mu = np.mean(data, axis=0)
    std = np.std(data, axis=0) + 1e-5

    return (data - mu) / std


def data_integration(event_full_path, label_dict, vocab, st_length):
    """
    Integrate data for dataset

    :param event_full_path: path of JSON file
    :param label_dict: dict for training or validation
    :param vocab: text_field
    :param st_length: fixed length of sentences
    :return: feature matrix of nodes, adjacency matrix, list of transformed word list, label
    """

    graph, eid, text_list = json_to_network(event_full_path, vocab, st_length)

    # label
    label = np.asarray(label_dict[eid], dtype=np.longlong)

    # adjacency matrix/feature matrix of nodes
    adj = np.asarray(nx.adjacency_matrix(graph).todense(), dtype=np.float32)
    node_info = np.asarray(get_node_info(graph), dtype=np.float32)
    node_info = feature_normalize(node_info)
    text_info = np.asarray(text_list, dtype=np.longlong)

    return node_info, adj, text_info, label
