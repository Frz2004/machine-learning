#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import re
import jieba.posseg as psg
import itertools


def Data_Cleansing():
    data = pd.read_excel("京东华为Pura 70 Ultra最新评论.xlsx")# 读取评论集Excel文件
    data = data[['评论内容', '评论打分']].drop_duplicates()  # 去重
    # data.to_excel('data_tmp.xlsx')
    content = data['评论内容']# 提取评论内容
    # 评论分数转换
    data.loc[data['评论打分'] == 1, '评论打分'] = 'neg'
    data.loc[data['评论打分'] == 3, '评论打分'] = 'neg'# 将评论分数为1和3的评论标记为负向评价（neg）
    data.loc[data['评论打分'] == 5, '评论打分'] = 'pos'# 将评论分数为5的评论标记为正向评价（pos）
    print(data.head())

    # 进行数据过滤

    # 去除去除英文、数字等
    # 由于评论主要为华为Pura 70 Ultra的评论，因此去除这些词语
    str_tmp = re.compile('[0-9a-zA-Z]|京东|手机|华为|华为Pura|华为Pura 70 Ultra|')  # re模块 正则表达式
    content = content.apply(lambda x: str_tmp.sub('', str(x)))  # 空值替换匹配内容

    # 进行评论分析

    # 评论拆分
    word_tmp = lambda s: [(x.word, x.flag) for x in psg.cut(s)]  # 自定义一个简单的拆分函数，将评论内容拆分成词语及其词性
    seg_word = content.apply(word_tmp)

    # 将词语转为数据框形式，一列是词，一列是词语所在的句子ID，最后一列是词语在该句子的位置
    n_word = seg_word.apply(lambda x: len(x))  # 计算每一评论中词的个数

    n_content = [[x + 1] * y for x, y in zip(list(seg_word.index), list(n_word))]
    index_content = sum(n_content, [])  # 将嵌套的列表展开，作为词所在评论的id

    seg_word = sum(seg_word, [])
    word = [x[0] for x in seg_word]  # 提取词

    nature = [x[1] for x in seg_word]  # 提取词性

    content_type = [[x] * y for x, y in zip(list(data['评论打分']), list(n_word))]
    content_type = sum(content_type, [])  # 初始化一个列表，用于存储评论类型

    # 构造结果数据框
    result = pd.DataFrame({"index_content": index_content,
                           "word": word,
                           "nature": nature,
                           "content_type": content_type})

    # 删除标点符号
    result = result[result['nature'] != 'x']  # x表示标点符号

    # 删除停用词
    # 读取停用词文件
    stop_path = open("./data/stoplist.txt", 'r', encoding='UTF-8')
    stop = stop_path.readlines()
    stop = [x.replace('\n', '') for x in stop]
    word = list(set(word) - set(stop))
    result = result[result['word'].isin(word)]

    # 构造各词在对应评论的位置列
    # 计算每个评论中的词的数量
    n_word = list(result.groupby(by=['index_content'])['index_content'].count())
    index_word = [list(np.arange(0, y)) for y in n_word]
    index_word = sum(index_word, [])  # 初始化一个列表，用于存储词语在该评论的位置

    # 合并评论id，评论中词的id，词，词性，评论打分
    result['index_word'] = index_word

    # 提取含有名词类的评论
    ind = result[['n' in x for x in result['nature']]]['index_content'].unique()
    result = result[[x in ind for x in result['index_content']]]# 初始化一个集合，用于存储含有名词类的评论的id

    # 绘制词云

    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    frequencies = result.groupby(by=['word'])['word'].count()
    frequencies = frequencies.sort_values(ascending=False)# 计算词频
    backgroud_Image = plt.imread('./data/pl.jpg')# 读取背景图片
    # 初始化WordCloud对象
    wordcloud = WordCloud(font_path="simsun.ttf",
                          max_words=100,
                          background_color='white',
                          mask=backgroud_Image)
    my_wordcloud = wordcloud.fit_words(frequencies)# 生成词云
    # 显示词云
    plt.imshow(my_wordcloud)
    plt.axis('off')
    plt.show()

    # 将结果写出
    result.to_excel("./tmp/word_shouji.xlsx", index=False)


def Commentary_Analysis():
    word = pd.read_excel("./tmp/word_shouji.xlsx")

    # 读入褒贬评价词
    pos_comment = pd.read_csv("./data/好词.txt", header=None, sep="\n",
                              encoding='utf-8', engine='python')
    neg_comment = pd.read_csv("./data/坏词.txt", header=None, sep="\n",
                              encoding='utf-8', engine='python')
    pos_emotion = pd.read_csv("./data/褒义.txt", header=None, sep="\n",
                              encoding='utf-8', engine='python')
    neg_emotion = pd.read_csv("./data/贬义.txt", header=None, sep="\n",
                              encoding='utf-8', engine='python')

    # 合并褒贬词与评价词
    positive = set(pos_comment.iloc[:, 0]) | set(pos_emotion.iloc[:, 0])
    negative = set(neg_comment.iloc[:, 0]) | set(neg_emotion.iloc[:, 0])
    intersection = positive & negative  # 褒贬表中相同的词语
    positive = list(positive - intersection)
    negative = list(negative - intersection)
    positive = pd.DataFrame({"word": positive,
                             "weight": [1] * len(positive)})
    negative = pd.DataFrame({"word": negative,
                             "weight": [-1] * len(negative)})

    posneg = positive.append(negative)

    #  将评论切割结果与褒贬表合并，定位关键词
    data_posneg = posneg.merge(word, left_on='word', right_on='word',
                               how='right')
    data_posneg = data_posneg.sort_values(by=['index_content', 'index_word'])

    # 修正倾向

    # 处理双重否定句
    # 载入否定词表
    notdict = pd.read_csv("./data/not.csv")

    # 处理否定修饰词
    data_posneg['amend_weight'] = data_posneg['weight']  # 构造经过否定修正后的新列
    data_posneg['id'] = np.arange(0, len(data_posneg))
    only_inclination = data_posneg.dropna()  # 只保留有评价值的词语
    only_inclination.index = np.arange(0, len(only_inclination))
    index = only_inclination['id']

    for i in np.arange(0, len(only_inclination)):
        review = data_posneg[data_posneg['index_content'] ==
                             only_inclination['index_content'][i]]  # 提取第i个关键词所在的评论
        review.index = np.arange(0, len(review))
        affective = only_inclination['index_word'][i]  # 第i个关键值在该文档的位置
        if affective == 1:
            ne = sum([i in notdict['term'] for i in review['word'][affective - 1]])
            if ne == 1:
                data_posneg['amend_weight'][index[i]] = - \
                    data_posneg['weight'][index[i]]
        elif affective > 1:
            ne = sum([i in notdict['term'] for i in review['word'][[affective - 1,
                                                                    affective - 2]]])
            if ne == 1:
                data_posneg['amend_weight'][index[i]] = - \
                    data_posneg['weight'][index[i]]

    # 更新只保留评价值的数据
    only_inclination = only_inclination.dropna()

    # 计算每条评论的评价值
    emotional_value = only_inclination.groupby(['index_content'],
                                               as_index=False)['amend_weight'].sum()

    # 去除评价值为0的评论
    emotional_value = emotional_value[emotional_value['amend_weight'] != 0]

    # 查看提取效果

    # 给评价值大于0的赋予评论类型（content_type）为pos,小于0的为neg
    emotional_value['a_type'] = ''
    emotional_value['a_type'][emotional_value['amend_weight'] > 0] = 'pos'
    emotional_value['a_type'][emotional_value['amend_weight'] < 0] = 'neg'

    # 查看评价分析结果
    result = emotional_value.merge(word,
                                   left_on='index_content',
                                   right_on='index_content',
                                   how='left')

    result = result[['index_content', 'content_type', 'a_type']].drop_duplicates()
    confusion_matrix = pd.crosstab(result['content_type'], result['a_type'],
                                   margins=True)  # 制作交叉表
    print((confusion_matrix.iat[0, 0] + confusion_matrix.iat[1, 1]) / confusion_matrix.iat[2, 2])
    print(confusion_matrix)
    # 提取正负面评论信息
    ind_pos = list(emotional_value[emotional_value['a_type'] == 'pos']['index_content'])
    ind_neg = list(emotional_value[emotional_value['a_type'] == 'neg']['index_content'])
    posdata = word[[i in ind_pos for i in word['index_content']]]
    negdata = word[[i in ind_neg for i in word['index_content']]]

    # 绘制词云
    import matplotlib.pyplot as plt
    from wordcloud import WordCloud

    # 正面关键词词云
    freq_pos = posdata.groupby(by=['word'])['word'].count()
    freq_pos = freq_pos.sort_values(ascending=False)
    backgroud_Image = plt.imread('./data/pl.jpg')
    wordcloud = WordCloud(font_path="simsun.ttf",
                          max_words=100,
                          background_color='white',
                          mask=backgroud_Image)
    pos_wordcloud = wordcloud.fit_words(freq_pos)
    plt.imshow(pos_wordcloud)
    plt.axis('off')
    plt.show()
    # 负面关键词词云
    freq_neg = negdata.groupby(by=['word'])['word'].count()
    freq_neg = freq_neg.sort_values(ascending=False)
    neg_wordcloud = wordcloud.fit_words(freq_neg)
    plt.imshow(neg_wordcloud)
    plt.axis('off')
    plt.show()

    # 将结果写出,每条评论作为一行
    posdata.to_excel("./tmp/posdata_shouji.xlsx", index=False)
    negdata.to_excel("./tmp/negdata_shouji.xlsx", index=False)


def LDA_Model():
    # 载入分析后的数据
    posdata = pd.read_excel("./tmp/posdata_shouji.xlsx")
    negdata = pd.read_excel("./tmp/negdata_shouji.xlsx")

    from gensim import corpora, models
    # 建立词典
    pos_dict = corpora.Dictionary([[i] for i in posdata['word']])  # 正面
    neg_dict = corpora.Dictionary([[i] for i in negdata['word']])  # 负面

    # 建立语料库
    pos_corpus = [pos_dict.doc2bow(j) for j in [[i] for i in posdata['word']]]  # 正面
    neg_corpus = [neg_dict.doc2bow(j) for j in [[i] for i in negdata['word']]]  # 负面

    # 深度分析部分

    # 构造LDA主题数寻优函数
    def cos(vector1, vector2):  # 余弦相似度函数
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_product += a * b
            normA += a ** 2
            normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return (None)
        else:
            return (dot_product / ((normA * normB) ** 0.5))

            # LDA

    def lda_k(x_corpus, x_dict):

        # 初始化平均余弦相似度
        mean_similarity = []
        mean_similarity.append(1)

        # 循环生成主题并计算主题间相似度
        for i in np.arange(2, 11):
            lda = models.LdaModel(x_corpus, num_topics=i, id2word=x_dict)  # LDA模型训练
            for j in np.arange(i):
                term = lda.show_topics(num_words=50)

            # 提取各主题词
            top_word = []
            for k in np.arange(i):
                top_word.append([''.join(re.findall('"(.*)"', i)) \
                                 for i in term[k][1].split('+')])  # 列出所有词

            # 构造词频向量
            word = sum(top_word, [])  # 列出所有的词
            unique_word = set(word)  # 去除重复的词

            # 构造主题词列表，行表示主题号，列表示各主题词
            mat = []
            for j in np.arange(i):
                top_w = top_word[j]
                mat.append(tuple([top_w.count(k) for k in unique_word]))

            p = list(itertools.permutations(list(np.arange(i)), 2))
            l = len(p)
            top_similarity = [0]
            for w in np.arange(l):
                vector1 = mat[p[w][0]]
                vector2 = mat[p[w][1]]
                top_similarity.append(cos(vector1, vector2))

            # 计算平均余弦相似度
            mean_similarity.append(sum(top_similarity) / l)
        return (mean_similarity)

    # 计算主题平均余弦相似度
    pos_k = lda_k(pos_corpus, pos_dict)
    neg_k = lda_k(neg_corpus, neg_dict)

    # 绘制主题平均余弦相似度图形
    from matplotlib.font_manager import FontProperties
    font = FontProperties(size=14)
    # 解决中文显示问题
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(211)
    ax1.plot(pos_k)
    ax1.set_xlabel('正面评论LDA主题数寻优', fontproperties=font)

    ax2 = fig.add_subplot(212)
    ax2.plot(neg_k)
    ax2.set_xlabel('负面评论LDA主题数寻优', fontproperties=font)
    plt.show()

    # LDA主题分析
    pos_lda = models.LdaModel(pos_corpus, num_topics=2, id2word=pos_dict)
    neg_lda = models.LdaModel(neg_corpus, num_topics=2, id2word=neg_dict)
    print(pos_lda.print_topics(num_words=10))

    print(neg_lda.print_topics(num_words=10))


if __name__ == '__main__':
    Data_Cleansing()
    Commentary_Analysis()
    LDA_Model()
