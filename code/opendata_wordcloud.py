# library
import pandas as pd
import matplotlib.pyplot as plot
from pyparsing import Word
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer

# dataset
d_2018 = pd.read_csv('data/2018_comment.csv')
d_2020 = pd.read_csv('data/2020_comment.csv')
d_pop = pd.read_csv('data/20220422_2015population.csv')

print(d_2018.shape)
print(d_2020.shape)

# data handle
d = pd.concat([d_2018, d_2020], axis='index')

df = pd.merge(d, d_pop, on='code', how='left')
print(df.head())
print(df.tail())

# 型はseriesというpandasの型になってるみたい。ベクトルではない...?
txt2018 = d_2018.comment
txt2020 = d_2020.comment
txt = df.comment

# 形態素解析のための関数
FONT_PATH = './data/07YasashisaAntique.ttf'
def get_word_str(text):
    from janome.tokenizer import Tokenizer
    import re
 
    t = Tokenizer()
    token = t.tokenize(text)
    word_list = []
 
    for line in token:
        tmp = re.split('\t|,', str(line))
        # 対象: 名詞と形容詞
        if tmp[1] in ["名詞"]:
            word_list.append(tmp[0])
        elif tmp[1] in ["形容詞"]:
            word_list.append(tmp[0])
    
    return " " . join(word_list)

# 1回答ごとにwordcloudを作成してしまうので、seriesをリストに変換し、リストをjoin()関数で結合する
## 2018
txt2018_list = txt2018.values.tolist()
txt2018_merge = ' '.join(txt2018_list)
## 2020
txt2020_list = txt2020.values.tolist()
txt2020_merge = ' '.join(txt2020_list)
## merge
txt_list = txt.values.tolist()
txt_merge = ' '.join(txt_list)

# 名詞と形容詞を集める
word_str = get_word_str(txt_merge)

stop_words = ['自治体', '愛知', '名古屋', '県', '市', '区', '都道府県', '市町村', 'こと', 'もの', '等', '化', 'よう', 'ため', 'の', 'ところ']

wc = WordCloud(font_path=FONT_PATH,
    stopwords = set(stop_words),
    max_font_size = 200, 
    background_color="white",
    collocations = False,
    width=800,height=600).generate(word_str)

wc.to_file("./output/wordcloud_merge.png")

## 2018
word_str = get_word_str(txt2018_merge)
wc = WordCloud(font_path=FONT_PATH,
    stopwords = set(stop_words),
    max_font_size = 200, 
    background_color="white",
    collocations = False,
    width=800,height=600).generate(word_str)
wc.to_file("./output/wordcloud_2018.png")

## 2020
word_str = get_word_str(txt2020_merge)
wc = WordCloud(font_path=FONT_PATH,
    stopwords = set(stop_words),
    max_font_size = 200, 
    background_color="white",
    collocations = False,
    width=800,height=600).generate(word_str)
wc.to_file("./output/wordcloud_2020.png")