import argparse
from data_load import *
from gat import *
import gc
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_(train_iter, val_iter, model, learning_rate, epochs, interval, batch_size, train_size, val_size):
    """
    Training function

    :param train_iter: iteration of training data
    :param val_iter: iteration of validation data
    :param model: GAT
    :param learning_rate: fixed learning rate (it should be dynamically adjusted during actual training)
    :param epochs: fixed epochs (no specific value would be set during actual training)
    :param interval: print training result when cumulative batch could be divided by interval
    :param batch_size: size of a batch
    :param train_size: size of training data
    :param val_size: size of validation data
    :return: loss of training and validation, accuracy of training and validation
    """

    model = model.to(DEVICE)
    Loss, val_Loss = [], []
    Acc, val_Acc = [], []
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        i = 0
        model.train()
        print("Epoch: ", epoch)
        correct_count = 0
        loss_sum = 0

        for node_info, adj, text_info, label in train_iter:
            torch.cuda.empty_cache()
            opt.zero_grad()
            predict, feature_vec = model(node_info, adj, text_info)
            loss = F.cross_entropy(predict, label)
            loss.backward()
            opt.step()
            correct_count += (torch.max(predict, 1)[1].view(label.size()) == label).sum().item()
            loss_sum += loss.item()
            i += batch_size
            if i % interval == 0:
                print("Epoch: {} {}/{}\t Current_mean_acc: {:.6f}\t Current_mean_loss: {:.6f}".format(
                    epoch, i, train_size, correct_count / i, loss_sum / i))
            del node_info, adj, text_info, label, feature_vec
            gc.collect()

        epoch_mean_loss = loss_sum / train_size
        epoch_mean_acc = correct_count / train_size
        print("Finishing train epoch: {}\t Accuracy: {:.6f}\t Loss: {:.6f}".format(
            epoch, epoch_mean_acc, epoch_mean_loss
        ))
        Loss.append(epoch_mean_loss)
        Acc.append(epoch_mean_acc)

        val_loss, val_acc = eval_(epoch, val_iter, model, val_size)
        val_Loss.append(val_loss)
        val_Acc.append(val_acc)

    return Loss, Acc, val_Loss, val_Acc


def eval_(epoch, val_iter, model, val_size):
    """
    Validation function

    :param epoch: current epoch
    :param val_iter: iteration of validation data
    :param model: GAT
    :param val_size: size of validation data
    :return: loss and accuracy of validation
    """
    model.eval()

    with torch.no_grad():
        correct_count = 0
        loss_sum = 0

        for node_info, adj, text_info, label in val_iter:
            torch.cuda.empty_cache()
            predict, feature_vec = model(node_info, adj, text_info)
            loss_sum += F.cross_entropy(predict, label).item()
            correct_count += (torch.max(predict, 1)[1].view(label.size()) == label).sum().item()
            del node_info, adj, text_info, label, feature_vec
            gc.collect()

        mean_loss = loss_sum / val_size
        mean_acc = correct_count / val_size
        print("Epoch val result: {}\t Accuracy: {:.6f}\t Loss: {:.6f}".format(
            epoch, mean_acc, mean_loss
        ))

    return mean_loss, mean_acc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo of GAT for fake news detection")
    parser.add_argument("-wtp", "--weibo_txt_path", default="sample_dataset/Weibo.txt", help="path of weibo_txt")
    parser.add_argument("-r", "--rate", default=0.8, help="rate of data for training")
    parser.add_argument("-jfbp", "--json_file_base_path", default="sample_dataset/Weibo",
                        help="base path of JSON file")
    parser.add_argument("-sl", "--sentence_length", default=50, help="fixed length of sentence")
    parser.add_argument("-bs", "--batch_size", default=2, help="size of a batch")
    parser.add_argument("-nf", "--n_feat", default=17,
                        help="feature dim of input layer (post_feature_size + poster_feature_size + output_size)")
    parser.add_argument("-nh1", "--n_hid_1", default=10, help="feature dim of the first hidden layer")
    parser.add_argument("-nh2", "--n_hid_2", default=10, help="feature dim of the second hidden layer")
    parser.add_argument("-nc", "--n_class", default=2, help="class num")
    parser.add_argument("-dpo", "--dropout", default=0.1, help="dropout rate")
    parser.add_argument("-alp", "--alpha", default=0.1, help="parameter alpha of function leakyrelu")
    parser.add_argument("-ed", "--embed_dim", default=16, help="feature dim of embedding layer")
    parser.add_argument("-fd", "--filter_dim", default=4, help="filter num of convs (TextCNNLayer)")
    parser.add_argument("-fs", "--filter_size", default=[2, 3, 4], help="filter size of convs (TextCNNLayer)")
    parser.add_argument("-os", "--output_size", default=6, help="filter size of convs (TextCNNLayer)")
    parser.add_argument("-lr", "--learning_rate", default=0.01, help="learning rate")
    parser.add_argument("-epc", "--epoch", default=10, help="fixed epoch")
    parser.add_argument("-itv", "--interval", default=2,
                        help="print training result when cumulative batch could be divided by interval")
    args = parser.parse_args()

    weibo_txt_path = args.weibo_txt_path
    rate = args.rate
    json_file_base_path = args.json_file_base_path
    sentence_length = args.sentence_length
    batch_size = args.batch_size
    n_feat = args.n_feat
    n_hid_1 = args.n_hid_1
    n_hid_2 = args.n_hid_2
    n_class = args.n_class
    dropout = args.dropout
    alpha = args.alpha
    embed_dim = args.embed_dim
    filter_dim = args.filter_dim
    filter_size = args.filter_size
    output_size = args.output_size
    learn_rate = args.learning_rate
    epoch = args.epoch
    itv = args.interval

    print("---------------Constructing Dataloader---------------")
    train_ld, val_ld = construct_label_dict(weibo_txt_path, rate)
    tl = text_collect(json_file_base_path, [i[0] for i in train_ld.items()])
    text = vocab_build(tl, sentence_length)
    train_dataset = WeiboDataset(json_file_base_path, [i[0] for i in train_ld.items()], train_ld, text, sentence_length)
    val_dataset = WeiboDataset(json_file_base_path, [i[0] for i in val_ld.items()], val_ld, text, sentence_length)
    weibo_train_loader = data.DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_)
    weibo_val_loader = data.DataLoader(val_dataset, shuffle=False, batch_size=batch_size, collate_fn=collate_)

    print("---------------Training data---------------")
    model = TextCNNGAT(n_feat, n_hid_1, n_hid_2, n_class, dropout, alpha, len(text.vocab), embed_dim,
                       filter_dim, filter_size, output_size, sentence_length)
    Loss, Acc, val_Loss, val_Acc = train_(weibo_train_loader, weibo_val_loader, model, learn_rate, epoch, itv,
                                          batch_size, len(train_ld), len(val_ld))
