import argparse
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score, roc_curve
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

np.random.seed(2020)

def read_data_file():
    with open("log_var.pkl", "rb") as f:
        log_var = pickle.load(f)
    with open("y_true.pkl", "rb") as f:
        y_true = pickle.load(f)
    with open("y_pred.pkl", "rb") as f:
        y_pred = pickle.load(f)
    return log_var, y_true, y_pred

def compute_accuracy_score(log_var, y_true, y_pred):
    idx_log_var = np.argsort(-np.array(log_var))
    idx_oracle = np.argsort(-np.square((np.array(y_true) - np.array(y_pred))))
    idx_random = [i for i in range(len(y_true))]
    np.random.shuffle(idx_random)

    pred_label = 1.0 * (np.array(y_pred) > 0.5)
    pred_label_oracle = np.array(pred_label)
    pred_label_random = np.array(pred_label)

    score = []
    oracle_score = []
    random_score = []
    ''' MSE
    for i in idx_log_var:
        y_pred[i] = y_true[i]
        score.append(np.sum(2 * np.square((np.array(y_true) - np.array(y_pred)))) / len(y_true))
    '''

    for i in idx_log_var:
        pred_label[i] = y_true[i]
        score.append(accuracy_score(y_true, pred_label))
    for i in idx_oracle:
        pred_label_oracle[i] = y_true[i]
        oracle_score.append(accuracy_score(y_true, pred_label_oracle))
    for i in idx_random:
        pred_label_random[i] = y_true[i]
        random_score.append(accuracy_score(y_true, pred_label_random))

    query_rate = [100.0 * (i + 1) / (len(score)) for i in range(len(score))]
    score = [100.0 * s for s in score]
    oracle_score = [100.0 * s for s in oracle_score]
    random_score = [100.0 * s for s in random_score]
    return query_rate, score, oracle_score, random_score

def draw_graph(query_rate, pred_score, oracle_score, random_score, title, x_title, y_title):
    plt.tight_layout()
    plt.figure()
    plt.title(title, fontsize=16)
    plt.xlabel(x_title, fontsize=16)
    plt.ylabel(y_title, fontsize=16)
    plt.tick_params(labelsize=14)
    plt.plot(query_rate, oracle_score, label="Oracle", linewidth=2.0)
    plt.plot(query_rate, pred_score, label="Ours", linewidth=2.0)
    plt.plot(query_rate, random_score, label="Random", linewidth=2.0)
    plt.legend(fontsize=16)
    #plt.show()
    plt.savefig(args.metric + ".png", format="png", bbox_inches="tight")
    plt.close()
    '''
    img = mpimg.imread(args.metric + ".png")
    plt.imshow(img)
    plt.show()
    '''

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="loss",
                        help="ks, auc, ap, loss")
    return parser.parse_args()


def compute_auc(log_var, y_true, y_pred):
    idx_log_var = np.argsort(-np.array(log_var))
    idx_oracle = np.argsort(-np.square((np.array(y_true) - np.array(y_pred))))
    idx_random = [i for i in range(len(y_true))]
    np.random.shuffle(idx_random)

    pred_label = np.array(y_pred)
    pred_label_oracle = np.array(pred_label)
    pred_label_random = np.array(pred_label)

    score = []
    oracle_score = []
    random_score = []
    ''' MSE
    for i in idx_log_var:
        y_pred[i] = y_true[i]
        score.append(np.sum(2 * np.square((np.array(y_true) - np.array(y_pred)))) / len(y_true))
    '''

    for i in idx_log_var:
        pred_label[i] = y_true[i]
        score.append(roc_auc_score(y_true, pred_label))
    for i in idx_oracle:
        pred_label_oracle[i] = y_true[i]
        oracle_score.append(roc_auc_score(y_true, pred_label_oracle))
    for i in idx_random:
        pred_label_random[i] = y_true[i]
        random_score.append(roc_auc_score(y_true, pred_label_random))

    query_rate = [100.0 * (i + 1) / (len(score)) for i in range(len(score))]
    score = [100.0 * s for s in score]
    oracle_score = [100.0 * s for s in oracle_score]
    random_score = [100.0 * s for s in random_score]
    return query_rate, score, oracle_score, random_score


def compute_ap(log_var, y_true, y_pred):
    idx_log_var = np.argsort(-np.array(log_var))
    idx_oracle = np.argsort(-np.square((np.array(y_true) - np.array(y_pred))))
    idx_random = [i for i in range(len(y_true))]
    np.random.shuffle(idx_random)

    pred_label = np.array(y_pred)
    pred_label_oracle = np.array(pred_label)
    pred_label_random = np.array(pred_label)

    score = []
    oracle_score = []
    random_score = []
    ''' MSE
    for i in idx_log_var:
        y_pred[i] = y_true[i]
        score.append(np.sum(2 * np.square((np.array(y_true) - np.array(y_pred)))) / len(y_true))
    '''

    for i in idx_log_var:
        pred_label[i] = y_true[i]
        score.append(average_precision_score(y_true, pred_label))
    for i in idx_oracle:
        pred_label_oracle[i] = y_true[i]
        oracle_score.append(average_precision_score(y_true, pred_label_oracle))
    for i in idx_random:
        pred_label_random[i] = y_true[i]
        random_score.append(average_precision_score(y_true, pred_label_random))

    query_rate = [100.0 * (i + 1) / (len(score)) for i in range(len(score))]
    score = [100.0 * s for s in score]
    oracle_score = [100.0 * s for s in oracle_score]
    random_score = [100.0 * s for s in random_score]
    return query_rate, score, oracle_score, random_score


def compute_loss(log_var, y_true, y_pred):
    idx_log_var = np.argsort(-np.array(log_var))
    idx_oracle = np.argsort(-np.square((np.array(y_true) - np.array(y_pred))))
    idx_random = [i for i in range(len(y_true))]
    np.random.shuffle(idx_random)

    pred_label = np.array(y_pred)
    pred_label_oracle = np.array(pred_label)
    pred_label_random = np.array(pred_label)

    score = []
    oracle_score = []
    random_score = []
    ''' MSE
    for i in idx_log_var:
        y_pred[i] = y_true[i]
        score.append(np.sum(2 * np.square((np.array(y_true) - np.array(y_pred)))) / len(y_true))
    '''

    for i in idx_log_var:
        pred_label[i] = y_true[i]
        score.append(np.sum(2 * np.square((np.array(y_true) - np.array(pred_label)))) / len(y_true))
    for i in idx_oracle:
        pred_label_oracle[i] = y_true[i]
        oracle_score.append(np.sum(2 * np.square((np.array(y_true) - np.array(pred_label_oracle)))) / len(y_true))
    for i in idx_random:
        pred_label_random[i] = y_true[i]
        random_score.append(np.sum(2 * np.square((np.array(y_true) - np.array(pred_label_random)))) / len(y_true))

    query_rate = [100.0 * (i + 1) / (len(score)) for i in range(len(score))]
    score = [100.0 * s for s in score]
    oracle_score = [100.0 * s for s in oracle_score]
    random_score = [100.0 * s for s in random_score]
    return query_rate, score, oracle_score, random_score


def ks(y_true, pred_label):
    fpr, tpr, ths = roc_curve(np.array(y_true).flatten(), pred_label.flatten())
    return max(tpr-fpr)


def compute_ks(log_var, y_true, y_pred):
    idx_log_var = np.argsort(-np.array(log_var))
    idx_oracle = np.argsort(-np.square((np.array(y_true) - np.array(y_pred))))
    idx_random = [i for i in range(len(y_true))]
    np.random.shuffle(idx_random)

    pred_label = np.array(y_pred)
    pred_label_oracle = np.array(pred_label)
    pred_label_random = np.array(pred_label)

    score = []
    oracle_score = []
    random_score = []
    ''' MSE
    for i in idx_log_var:
        y_pred[i] = y_true[i]
        score.append(np.sum(2 * np.square((np.array(y_true) - np.array(y_pred)))) / len(y_true))
    '''

    for i in idx_log_var:
        pred_label[i] = y_true[i]
        score.append(ks(y_true, pred_label))
    for i in idx_oracle:
        pred_label_oracle[i] = y_true[i]
        oracle_score.append(ks(y_true, pred_label_oracle))
    for i in idx_random:
        pred_label_random[i] = y_true[i]
        random_score.append(ks(y_true, pred_label_random))


    query_rate = [100.0 * (i + 1) / (len(score)) for i in range(len(score))]
    score = [100.0 * s for s in score]
    oracle_score = [100.0 * s for s in oracle_score]
    random_score = [100.0 * s for s in random_score]

    return query_rate, score, oracle_score, random_score


if __name__ == '__main__':
    log_var, y_true, y_pred = read_data_file()
    title = None
    args = init_arg()
    if args.metric == "ks":
        query_rate, pred_score, oracle_score, random_score = compute_ks(log_var, y_true, y_pred)
        y_title = "KS Value (%)"
        x_title = "Query Rate (%)"
        draw_graph(query_rate, pred_score, oracle_score, random_score, title, x_title, y_title)
    elif args.metric == "auc":
        query_rate, pred_score, oracle_score, random_score = compute_auc(log_var, y_true, y_pred)
        y_title = "AUC-ROC (%)"
        x_title = "Query Rate (%)"
        draw_graph(query_rate, pred_score, oracle_score, random_score, title, x_title, y_title)
    elif args.metric == "ap":
        query_rate, pred_score, oracle_score, random_score = compute_ap(log_var, y_true, y_pred)
        y_title = "Average Precision (%)"
        x_title = "Query Rate (%)"
        draw_graph(query_rate, pred_score, oracle_score, random_score, title, x_title, y_title)
    elif args.metric == "loss":
        query_rate, pred_score, oracle_score, random_score = compute_loss(log_var, y_true, y_pred)
        y_title = "Predictive Loss (x1e-2)"
        x_title = "Query Rate (%)"
        draw_graph(query_rate, pred_score, oracle_score, random_score, title, x_title, y_title)
    else:
        print("Input an incorrect argument!")