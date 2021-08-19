from math import log 
import re
import pickle
import platform
import argparse

from sentence_transformers import SentenceTransformer
import umap
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import ParameterGrid
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import seaborn as sns
import pandas as pd 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import requests
from bs4 import BeautifulSoup as bs
from konlpy.tag import Okt
import hanja

if platform.system() == 'Darwin': #맥
        plt.rc('font', family='AppleGothic') 
elif platform.system() == 'Windows': #윈도우
        plt.rc('font', family='Malgun Gothic') 
elif platform.system() == 'Linux': #리눅스 (구글 콜랩)
        plt.rc('font', family='Malgun Gothic')

stop_words = {"'", '"', ',', '.', '/', ';', ':', '[', ']', '{', '}', '\\', '|', '`', '~' \
              , '!', '@', '#', '^', '&', '*', '(', ')', '-', '_', '+', '=', ' '}

def tf(t, d):
    res = 0
    for word in d:
        if t == word:
            res += 1
    return res

def idf(t, docs, N):
    df = 0
    for doc in docs:
        df += t in doc
    return log(N/(df + 1))

def tfidf(t, d, docs):
    return tf(t,d)* idf(t, docs)



def main(args):    
    # headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    # tokenizer = Okt()
    # news = set()
    # times = []
    # page_number = 1
    # while True:
    #     url = f'https://news.daum.net/breakingnews/politics?page={page_number}'

    #     res = requests.get(url, headers=headers)
    #     res.raise_for_status()

    #     if res.status_code == 200:
    #         html = bs(res.text, 'html.parser')
        
    #         title_list = html.select('#mArticle > div.box_etc > ul > li > div > strong')
    #         if not title_list: ## 마지막 페이지를 넘어간 경우 title_list는 빈 리스트가 된다.
    #             break
    #         for title in title_list:
    #             text = title.text.split('\n')[1]
    #             time = title.text.split('\n')[2][-5:-3]
    #             text = re.sub("\[.*\]|\{.*\}|\(.*\)",'',text) ## 괄호제거. 가끔 같은 제목에 괄호만 추가된 다른기사도 있음. 중복임.
    #             text2 = tokenizer.nouns(text) ## 명사가 없는 것은 쓰지 않는다.
    #             if len(text) > 10 and text2 and text not in news: 
    #                 news.add(text)
    #                 times.append(time)

    #     page_number += 1
    # news = list(news)
    
    # news = pickle.load(open('news_21_07_29.pickle', 'rb'))
    # times = pickle.load(open('times_21_07_29.pickle', 'rb'))
    # news = list(news)
    # news.sort()
    ## 한자를 한글로 바꾸기
    for i in range(len(news)):
        news[i] = hanja.translate(news[i], 'substitution')


    ## SBERT 임베딩
    embedder = SentenceTransformer(args.model_name)
    corpus_embeddings = embedder.encode(news)

    ## UMAP으로 차원 축소
    umap_embeddings = umap.UMAP(n_neighbors=args.n_neighbors,
                                n_components=args.n_components ,random_state=args.random_state, min_dist = 0,
                                metric='cosine').fit_transform(corpus_embeddings)
    # umap_embeddings = corpus_embeddings
    ## k-means로 클러스터링 하기 전에 스케일링
    se = StandardScaler()
    umap_embeddings = se.fit_transform(umap_embeddings)

    ## k-means의 하이퍼 파라미터(k) 서치. 실루엣 스코어 적용
    num_clusters_search = list(range(args.k_start, args.k_stop, args.k_step))
    parameter_grid = ParameterGrid({'n_clusters': num_clusters_search})
    best_score = -1
    silhouette_scores = []
    for p in parameter_grid:
        clustering_model = KMeans(random_state=args.random_state)
        clustering_model.set_params(**p)
        clustering_model.fit(umap_embeddings)
        ss = metrics.silhouette_score(umap_embeddings, clustering_model.labels_)
        silhouette_scores += [ss]
        if ss > best_score:
            best_score = ss
            best_grid = p

    ## 실루엣 스코어가 가장 클 때의 k로 학습
    num_clusters = best_grid['n_clusters']
    clustering_model = KMeans(n_clusters=num_clusters,random_state=args.random_state)
    clustering_model.fit(umap_embeddings)
    cluster_assignment = clustering_model.labels_

    ## 같은 클러스터 안의 뉴스기사를 하나의 리스트안에 넣음
    clustered_sentences = [[] for i in range(num_clusters)]
    clustered_times = [[] for i in range(num_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(news[sentence_id])
        clustered_times[cluster_id].append(times[sentence_id])

    ## 명사만 추출.
    tokenizer = Okt()
    tokenized_corpus = [[] for i in range(num_clusters)]
    for cluster_num in range(num_clusters):
        for sentence in clustered_sentences[cluster_num]:
            tokenized_corpus[cluster_num].append(tokenizer.nouns(sentence))

    ## 각 클러스터의 tfidf 계산.
    cluster_tfidf = []
    # N_tfidf_pre = []
    for cluster_num in range(len(tokenized_corpus)):
        docs = tokenized_corpus[cluster_num]
        vocab = list(set(w for doc in docs for w in doc))
        N = len(docs)
        
        sentence_num_vocab = []
        for sen in docs:
            sentence_num_vocab.append(len(sen))

        result = []
        for i in range(N): 
            result.append([])
            d = docs[i]
            for j in range(len(vocab)):
                t = vocab[j]        
                result[-1].append(tf(t, d))

        tf_ = pd.DataFrame(result, columns = vocab)

        result = []
        for j in range(len(vocab)):
            t = vocab[j]
            result.append(idf(t,docs,N))

        idf_ = pd.DataFrame(result, index = vocab, columns = ["IDF"])
        tfidf_ = pd.DataFrame(np.multiply(tf_.to_numpy(),idf_.to_numpy().T), columns = vocab)
        sentence_tfidf_sum = tfidf_.sum(axis=1).to_numpy()
        sentence_num_vocab = np.array(sentence_num_vocab)
        mean_tfidf = (sentence_tfidf_sum/ sentence_num_vocab).mean()

        cluster_tfidf.append([mean_tfidf, N ,list(idf_.sort_values(by='IDF').index[:4]), cluster_num])
        # N_tfidf_pre.append([N,mean_tfidf])

    ## x = log(클러스터 크기), y = 클러스터 tfidf 점수 로 선형회귀 학습.
    N_tfidf_pre = pickle.load(open('N_tfidf_pre.pickle', 'rb'))
    reg_log = LinearRegression().fit(np.log(N_tfidf_pre[:,0]).reshape(-1,1), N_tfidf_pre[:,1])
  
    ## 각 클러스터에 대하여 회귀한 함수의 tfidf값보다 작은 클러스터 들만 사용한다.
    good_cluster = []
    for e in cluster_tfidf:
        predicted_value = reg_log.predict(np.log( [[e[1]]] ))*0.95
        if e[0] <= predicted_value:
            good_cluster.append(e)
    good_cluster

    ## jacard simirality가 0.3 이상이면 같은 클러스터로 판단하고 merge.
    already_merged = set()
    new_clusters = []
    for i in range(len(good_cluster)):
        temp_clusters = []
        for j in range(i,len(good_cluster)):
            temp_i = set(good_cluster[i][2])
            temp_j = set(good_cluster[j][2])
            if good_cluster[j][3] not in already_merged and (len(temp_i&temp_j)/len(temp_i|temp_j))>=args.jacard_threshold:
                temp_clusters.append(good_cluster[j][3])
                already_merged.add(good_cluster[j][3])
        if temp_clusters:
            new_clusters.append(temp_clusters)

    new_clustered_sentences = [[] for i in range(len(new_clusters))]
    new_tokenized_corpus = [[] for i in range(len(new_clusters))]
    new_times = [[] for i in range(len(new_clusters))]
    for i in range(len(new_clusters)):
        for cluster_num in new_clusters[i]:
            new_clustered_sentences[i] += clustered_sentences[cluster_num]
            new_tokenized_corpus[i] += tokenized_corpus[cluster_num]
            new_times[i] += clustered_times[cluster_num]

    ## 새로운 클러스터를 직접 확인(주피터 노트북에서 확인하기).
    # for i, cluster in enumerate(new_clustered_sentences):
    #     print("Cluster ", i)
    #     print(cluster)
    #     print("")

    # for i, cluster in enumerate(new_tokenized_corpus):
    #     print("Cluster ", i)
    #     print(cluster)
    #     print("")

    ## merge된 클러스터에 대해서 다시 tfidf계산
    cluster_tfidf = []
    for cluster_num in range(len(new_tokenized_corpus)):
        docs = new_tokenized_corpus[cluster_num]
        vocab = list(set(w for doc in docs for w in doc))
        N = len(docs)
        
        sentence_num_vocab = []
        for sen in docs:
            sentence_num_vocab.append(len(sen))
        
        result = []
        for i in range(N): 
            result.append([])
            d = docs[i]
            for j in range(len(vocab)):
                t = vocab[j]        
                result[-1].append(tf(t, d))

        tf_ = pd.DataFrame(result, columns = vocab)

        result = []
        for j in range(len(vocab)):
            t = vocab[j]
            result.append(idf(t,docs,N))
           
        idf_ = pd.DataFrame(result, index = vocab, columns = ["IDF"])
        
        tfidf_ = pd.DataFrame(np.multiply(tf_.to_numpy(),idf_.to_numpy().T), columns = vocab)
        sentence_tfidf_sum = tfidf_.sum(axis=1).to_numpy()
        sentence_num_vocab = np.array(sentence_num_vocab)
        mean_tfidf = (sentence_tfidf_sum/ sentence_num_vocab).mean()  #tfidf_.apply(row_mean,axis=1).mean()
    
        cluster_tfidf.append([mean_tfidf, N ,list(idf_.sort_values(by='IDF').index[:4]), cluster_num])



    cluster_tfidf.sort(key=lambda x : x[1], reverse=True)

    ## 크기가 가장큰 n개의 클러스터의 title을 뽑는다.
    final_title = []
    num_c = min(args.num_final_clusters, len(cluster_tfidf))
    for i in range(num_c):
        cluster_size = cluster_tfidf[i][1]
        cluster_num = cluster_tfidf[i][3]
        top_tfidf_words = cluster_tfidf[i][2]
        num_tfidf_words = len(top_tfidf_words)
        score = 0
        num_title_tokens = float('inf')
        
        for i, tokens in enumerate(new_tokenized_corpus[cluster_num]):
            set_tokens = set(tokens)
            score_temp = 0
            for j,word in enumerate(top_tfidf_words):
                if word in set_tokens:
                    score_temp += (num_tfidf_words - j)
            if score_temp > score:
                score = score_temp
                num_title_tokens = len(tokens)
                cluster_title = new_clustered_sentences[cluster_num][i]
            elif score_temp == score and len(tokens) < num_title_tokens:
                num_title_tokens = len(tokens)
                cluster_title = new_clustered_sentences[cluster_num][i]
        indices = []        
        for token in top_tfidf_words:
            s_index = cluster_title.find(token)
            if s_index != -1:
                e_index = s_index + len(token) -1
                while (s_index-1)>=0 and cluster_title[s_index-1] not in stop_words:
                    s_index -= 1
                while (e_index+1) < len(cluster_title) and cluster_title[e_index+1] not in stop_words:
                    e_index += 1
                indices.append((s_index, e_index))
        indices = list(set(indices)) ## 중복제거
        indices.sort(key = lambda x : x[0])
        str_temp = []
        for e in indices:
            str_temp.append(cluster_title[e[0]:e[1]+1])
        final_title.append((' '.join(str_temp), cluster_num, cluster_size))

    ## 시각화
    column_order = []
    for _,i,_ in final_title:
        column_order.append(i)

    df_time = pd.DataFrame()
    for _, cluster_num, _ in final_title:
        df_time = pd.concat([df_time, pd.DataFrame({'time':new_times[cluster_num], 'cluster_num':[cluster_num]*len(new_times[cluster_num])})])
    df_time2 = pd.pivot_table(df_time, index='time', columns='cluster_num', values='time', fill_value=0, aggfunc=len)
    df_time2 = df_time2[column_order]
    time_perc = df_time2.divide(df_time2.sum(axis=1), axis=0)


    fig, ax = plt.subplots(1, 2, figsize=(24, 7))
    columns = []
    for c in time_perc.columns:
        columns.append(time_perc[c])
    labels = []
    for e in final_title:
        labels.append(e[0] + f'({e[2]})')

    fontP = FontProperties()
    fontP.set_size('medium')
        


    ax[0].stackplot(time_perc.index,
                columns,
                labels=labels,
                colors=sns.color_palette('dark'),
                alpha=0.8)
    ax[0].set_xlabel('time(hour)')
    ax[0].set_ylabel('percent')
    ax[0].set_yticklabels(['0%','20%', '40%', '60%', '80%', '100%'])
    ax[0].set_title('topic ratio')

    df_time = df_time.sort_values(by='time')
    count = sns.countplot(x=df_time['time'] ,data=df_time,
                hue='cluster_num', palette=sns.color_palette('dark'),
                hue_order=column_order,
                ax=ax[1])
    ax[1].set_xlabel('time(hour)')
    ax[1].set_title('topic count')
    ax[1].legend(title='title(cluster size)', bbox_to_anchor=(1.05, 1), loc='upper left', prop=fontP, labels=labels)
    plt.subplots_adjust(right=0.8)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', type=str, default='distiluse-base-multilingual-cased-v1')
    parser.add_argument('--random_state', type=int, default = 42)
    parser.add_argument('--n_neighbors', type=int, default = 15, help = 'umap parameter. it controls how UMAP balances local versus global structure in the data.')
    parser.add_argument('--n_components', type=int, default = 10, help ='umap parameter. dimensionality of the reduced dimension space.')
    parser.add_argument('--k_start', type=int, default = 10, help = 'start value for k in k-means hyperparameter search.')
    parser.add_argument('--k_stop', type=int, default = 51, help = 'stop value for k in k-means hyperparameter search.')
    parser.add_argument('--k_step', type=int, default = 5, help = 'step value for k in k-means hyperparameter search.')
    parser.add_argument('--num_idfs', type=int, default = 4, help = 'number of idf values for each cluster.')
    parser.add_argument('--jacard_threshold', type=float, default = 0.3)
    parser.add_argument('--num_final_clusters', type=int, default = 5)
    parser.add_argument('--tfidf_threshold_coeff', type=float, default = 0.95)


    args = parser.parse_args()
    main(args)