# library
from pydoc import describe
import pandas as pd
import matplotlib.pyplot as plot
from pyparsing import Word
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
import nlplot
import mca
import japanize_matplotlib

# dataset
d_2018 = pd.read_csv('data/2018_comment.csv')
d_2020 = pd.read_csv('data/2020_comment.csv')
d_pop = pd.read_csv('data/20220426_2015population.csv')

# data handle
d = pd.concat([d_2018, d_2020], axis='index')

df = pd.merge(d, d_pop, on='code', how='left')
print(df.head())
print(df.tail())

# 都道府県の除外と人口のカテゴリ化
df_cities = df[~(df['code'] % 1000 == 0)]
plot.hist(df_cities.population)
plot.show()
print(df_cities.describe())
# 369obs, population-> 25%: 17000, 50: 58370, 75%: 158114
# Rのdplyr::mutateのcase_whenや、SQLのcase whenがない...だと...
## https://qiita.com/Hyperion13fleet/items/98c31744e66ac1fc1e9f

def pop_categorize(x):
    if  x < 17000:
        return 'small'
    elif x >= 17000 and x < 58370:
        return 'relatively small'
    elif x >= 58370 and x < 158114:
        return 'relatively large'
    else:
        return 'large'

df_cities['pop_cat'] = df_cities['population'].apply(pop_categorize)
print(df_cities.head())

# 人口サイズのカテゴリごとのコメント
'''
txt_small = df_cities[(df_cities['pop_cat']  == 'small')].comment
txt_relsmall = df_cities[(df_cities['pop_cat']  == 'relatively small')].comment
txt_rellarge = df_cities[(df_cities['pop_cat']  == 'relatively large')].comment
txt_large = df_cities[(df_cities['pop_cat']  == 'large')].comment
'''

# 頻出単語と人口のクロス集計表の作成
'''
txt = df_cities.comment
txt_list = txt.values.tolist()
txt_merge = ' '.join(txt_list)
'''

def series_to_text(seriesdata):
    text_list = seriesdata.values.tolist()
    return ' '.join(text_list)

txt_merge = series_to_text(df_cities.comment)
txt_small = series_to_text(df_cities[(df_cities['pop_cat']  == 'small')].comment)
txt_relsmall = series_to_text(df_cities[(df_cities['pop_cat']  == 'relatively small')].comment)
txt_rellarge = series_to_text(df_cities[(df_cities['pop_cat']  == 'relatively large')].comment)
txt_large = series_to_text(df_cities[(df_cities['pop_cat']  == 'large')].comment)

t = Tokenizer()

'''
# 1つずつ処理する場合のコード。次に関数化するための準備
#3---テキストを一行ずつ処理
word_txt = {}
lines = txt_merge.split("\r\n")
for line in lines:
    malist = t.tokenize(line)
    for w in malist:
        word = w.surface#4---単語情報の読込
        ps = w.part_of_speech #5---品詞情報の読込
        if ps.find('名詞') < 0: continue #6---名詞のカウント
        if not word in word_txt:
            word_txt[word] = 0
        word_txt[word] += 1 #7---カウント
#8---頻出単語の表示
keys = sorted(word_txt.items(), key=lambda x:x[1], reverse=True)

n_words = 50
for word, cnt in keys[:n_words]:
    print("{0}({1}) ".format(word,cnt), end="")

# print(keys)

freq_overall = pd.DataFrame(keys,
    columns=['term', 'overall'])
print(freq_overall.head())
'''

# テキストを入力すると、登場する名詞の頻度を集計してdataframeを返す関数を作る

def freq_count_df(textdata):
    t = Tokenizer()
    word_txt = {}

    lines = textdata.split("\r\n")

    for line in lines:
        malist = t.tokenize(line)
        for w in malist:
            word = w.surface #単語情報の読込
            ps = w.part_of_speech #品詞情報の読込
            if ps.find('名詞') < 0: continue #名詞
            if not word in word_txt:
                word_txt[word] = 0
            word_txt[word] += 1 #7---カウント
    
    keys = sorted(word_txt.items(), key=lambda x:x[1], reverse=True)
    return pd.DataFrame(keys,columns=['term', 'total count'])

df_freq_overall = freq_count_df(txt_merge)

df_freq_small = freq_count_df(txt_small)
df_freq_small.columns = ['term','small']

df_freq_relsmall = freq_count_df(txt_relsmall)
df_freq_relsmall.columns = ['term','relatively small']

df_freq_rellarge = freq_count_df(txt_rellarge)
df_freq_rellarge.columns = ['term','relatively large']

df_freq_large = freq_count_df(txt_large)
df_freq_large.columns = ['term','large']

# stopwordsの行を除外して、クロス集計表へ
stop_words = ['自治体', '愛知', '名古屋', '県', '市', '区', '都道府県', '市町村', 'こと', 'もの', '等', '化', 'よう', 'ため', 'の', 'ところ']
df_freq_overall = df_freq_overall[~(df_freq_overall['term'].isin(stop_words))]

n_words = 50
df_freq_overall = df_freq_overall[~(df_freq_overall['term'].isin(stop_words))]
df_freq_overall = df_freq_overall[0:n_words]

df_freq_merge = pd.merge(df_freq_overall, df_freq_small, on='term', how='left')
df_freq_merge = pd.merge(df_freq_merge, df_freq_relsmall, on='term', how='left')
df_freq_merge = pd.merge(df_freq_merge, df_freq_rellarge, on='term', how='left')
df_freq_merge = pd.merge(df_freq_merge, df_freq_large, on='term', how='left')

print(df_freq_merge.head())

# term列をindexにして、削除する（クロス集計表にしたい）
index_series = df_freq_merge.term
index_list = index_series.values.tolist()

df_freq_merge.index = index_list
print(df_freq_merge.head())

del df_freq_merge['term']
del df_freq_merge['total count']
print(df_freq_merge.head())

# いよいよコレスポンデンス分析へ
mca_counts = mca.MCA(df_freq_merge, benzecri=False)
rows = mca_counts.fs_r(N=2)
cols = mca_counts.fs_c(N=2)


fig, ax=plot.subplots(figsize=(8,8))

# インデックスの処理
ax.scatter(rows[:,0], rows[:,1], c='pink', marker='o', s=200)
labels = df_freq_merge.index.values
for label,x,y in zip(labels,rows[:,0],rows[:,1]):
        ax.annotate(label,xy = (x, y),fontsize=10)

# カラムの処理
ax.scatter(cols[:,0], cols[:,1], c='skyblue', marker='s', s=200)
labels = df_freq_merge.columns.values
for label,x,y in zip(labels,cols[:,0],cols[:,1]):
        ax.annotate(label,xy = (x, y),fontsize=20)

# 原点（0,0）を引く
ax.axhline(0, color='gray')
ax.axvline(0, color='gray')
fig.savefig("output/corresspondence.png")
