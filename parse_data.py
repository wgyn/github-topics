import bson
import re
import time

import sklearn
import sklearn.decomposition

def normalize_string(s):
    return re.sub('[^\w\s]', '', s.lower()).strip()

def tokeep(description):
    if description:
        return len(description) > 5
    else:
        return False

if __name__=='__main__':
    with open('sample_data/raw/repos.bson', 'r') as f:
        repos = bson.decode_all(f.read())
    repos = filter(lambda r: tokeep(r['description']), repos)

    descs = {r['id']: normalize_string(r['description']) for r in repos if tokeep(r['description'])}

    vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        max_df=0.95, min_df=5, stop_words='english', max_features=1000)
    counts = vectorizer.fit_transform(descs.values())

    time = t0
    nmf = sklearn.decomposition.NMF(n_components=50, random_state=23).fit(counts)
    feature_names = vectorizer.get_feature_names()
    print "%d seconds elapsed" % (time() - t0)

    n_top_words = 10
    for topic_idx, topic in enumerate(nmf.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
        print()

    topic_features = nmf.transform(counts)
