import pickle
from sklearn.model_selection import train_test_split

def build_dict(file, dump):
    id2word, word2id = {}, {}
    with open(file) as f:
        vocabs = f.readlines()
    idx = 0
    for word in vocabs:
        word = word.split("\n")[0]
        if word not in word2id:
            word2id[word] = idx
            id2word[idx] = word
            idx += 1

    print("id of s, /s, unk", word2id["<s>"], word2id["</s>"], word2id["<unk>"])
    print("len of word2id is ", len(word2id))
    print("len of id2word is ", len(id2word))
    vocab_id = {"word2id":word2id, "id2word":id2word}
    with open(dump, 'wb') as f:
        pickle.dump(vocab_id, f)
    print("vocab to id have been built!")


def sentence2id(corups, vocab_dict):
    with open(corups) as f:
        sentences = f.readlines()
        # without ' ' delete ' ' and '\n'
        sentences = [s.split() for s in sentences]

    with open(vocab_dict, 'rb') as f:
        word2id = pickle.load(f)["word2id"]
        print(word2id["<unk>"], word2id["<s>"], word2id["</s>"])

    id_sentences = []
    for sentence in sentences:
        id_sen = []
        for word in sentence:
            if word in word2id:
                id_sen.append(word2id[word])
            else:
                id_sen.append(word2id["<unk>"])

        id_sentences.append(id_sen)
    return id_sentences


def filter_len(source, target):
    assert len(source) == len(target)
    n = len(source)
    new_s, new_t = [], []
    for i in range(n):
        if (len(source[i]) <= 50 and len(target[i]) <= 50
            and len(source[i]) >= 3 and len(target[i]) >= 3):
            new_s.append(source[i])
            new_t.append(target[i])
    print("filtered out %d"%(n - len(new_s)))
    return new_s, new_t


def bucket(source, target, bucket_num=5):
    n = len(source)
    bucket_s = [[] for _ in range(bucket_num)]
    bucket_t = [[] for _ in range(bucket_num)]
    for i in range(n):
        idx = 4
        sl, tl = len(source[i]), len(target[i])
        if sl <= 10 and tl <= 10:
            idx = 0
        elif sl <= 20 and tl <= 20:
            idx = 1
        elif sl <= 30 and tl <= 30:
            idx = 2
        elif sl <= 40 and tl <= 40:
            idx = 3
        else:
            idx = 4
        bucket_s[idx].append(source[i])
        bucket_t[idx].append(target[i])

    new_s, new_t = [], []
    for i in range(bucket_num-1, -1,-1):
        print(bucket_s[i][0], bucket_t[i][0])
        print(bucket_s[i][-1], bucket_t[i][-1])
        new_s.extend(bucket_s[i])
        new_t.extend(bucket_t[i])
    print(n, len(new_s))
    return new_s, new_t



def split_corups(source_data, target_data, source_dict, target_dict):
    # not needed for our data. deprecated.
    print("converting corups into id representation")
    source_id = sentence2id(source_data, source_dict)
    print(source_id[0])
    target_id = sentence2id(target_data, target_dict)
    print(target_id[0])

    source_filter, target_filter = filter_len(source_id, target_id)
    train_source, val_source, train_target, val_target = train_test_split(source_filter, 
                                                                          target_filter,
                                                                          test_size=0.01, 
                                                                          random_state=42)

    print("saving splited data into pickle files")
    print("each pickle files is a list of id-represented sentence")
    print("sentences in train set = ", len(train_source))
    with open("train.id.en.pkl", 'wb') as f:
        pickle.dump(train_source, f)
    with open("train.id.de.pkl", 'wb') as f:
        pickle.dump(train_source, f)
    with open("val.id.en.pkl", 'wb') as f:
        pickle.dump(val_source, f)
    with open("val.id.de.pkl", 'wb') as f:
        pickle.dump(val_target, f)


def make_corups(source_data, target_data, source_dict, target_dict,
                source_out, target_out):
    print("converting corups into id representation")
    source_id = sentence2id(source_data, source_dict)
    print(source_id[0])
    target_id = sentence2id(target_data, target_dict)
    print(target_id[0])

    source_filter, target_filter = filter_len(source_id, target_id)
    # if you want to make data into buckets by length, delete '#'.
    #source_filter, target_filter = bucket(source_filter, target_filter)
    with open(source_out, "wb") as f:
        pickle.dump(source_filter, f)
    with open(target_out, "wb") as f:
        pickle.dump(target_filter, f)
    print("corups have been made")


if __name__ == "__main__":
    #build_dict("vocab.en", 'vocab_id.en.pkl')
    #build_dict("vocab.vi", 'vocab_id.vi.pkl')
    make_corups("train.en", "train.vi", "vocab_id.en.pkl",
                "vocab_id.vi.pkl", "trainb.en.pkl", "trainb.vi.pkl")
    # make_corups("tst2012.en", "tst2012.vi", "vocab_id.en.pkl",
    #             "vocab_id.vi.pkl", "tst2012.en.pkl", "tst2012.vi.pkl")
    # make_corups("tst2013.en", "tst2013.vi", "vocab_id.en.pkl",
    #             "vocab_id.vi.pkl", "tst2013.en.pkl", "tst2013.vi.pkl")
