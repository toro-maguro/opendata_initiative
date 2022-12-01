# library
import pandas as pd
import matplotlib.pyplot as plot
from pyparsing import Word
from wordcloud import WordCloud
from janome.tokenizer import Tokenizer
import nlplot
import mca

# dataset
d_2018 = pd.read_csv('data/2018_comment.csv')
d_2020 = pd.read_csv('data/2020_comment.csv')
d_pop = pd.read_csv('data/20220422_2015population.csv')
obama = open('data/obama20090120.txt').read()
trump = open('data/trump20170120.txt').read()
biden = open('data/biden20210120.txt').read()
speech = pd.read_csv('data/president_speech.csv')

# data handle
d = pd.concat([d_2018, d_2020], axis='index')
print(d.shape)

df = pd.merge(d, d_pop, on='code', how='left')
print(df.head())
print(df.tail())

print(df.shape)

txt = df.comment
print(type(txt))
print(txt[1])

# 単語分割 + 品詞の情報 p.066
t_sample = txt[1]
t = Tokenizer()
for token in t.tokenize(t_sample):
    print(token)

# 単語分割のみの場合
t = Tokenizer(wakati=True) 
for token in t.tokenize(t_sample):
    print(token)

# 形態素解析によって単語を分割し、スペースをはさむ必要あり
wordcloud = WordCloud(
    background_color="white",
    font_path='./data/07YasashisaAntique.ttf',
    width=400,height=300).generate(txt[2])
wordcloud.to_file("./output/wordcloud_sample.png")
# コンテナ環境で分析しているので、ローカルのフォントデータにアクセスできないことに注意
# https://blog.vtryo.me/entry/parse-messages-with-wordcloud

obama_wc = WordCloud(
    background_color="white",
    width=400, height=300,
    max_font_size=100).generate(obama)
obama_wc.to_file("./output/wordcloud_sample_obama.png")



# 形態素解析やろう
FONT_PATH = './data/07YasashisaAntique.ttf'

def get_word_str(text):
    from janome.tokenizer import Tokenizer
    import re
 
    t = Tokenizer()
    token = t.tokenize(text)
    word_list = []
 
    for line in token:
        tmp = re.split('\t|,', str(line))
        # 名詞のみ対象
        if tmp[1] in ["名詞"]:
            # さらに絞り込み
            if tmp[2] in ["一般", "固有名詞"]:
                word_list.append(tmp[0])
    
    return " " . join(word_list)



word_str = get_word_str(txt[2])
wc = WordCloud(font_path=FONT_PATH,
    max_font_size = 40, 
    background_color="white",
    width=800,height=600).generate(word_str)

wc.to_file("./output/wordcloud_sample_str.png")

# 1回答ごとにwordcloudを作成してしまうので、seriesをリストに変換し、リストをjoin()関数で結合する
txt_list = txt.values.tolist()
txt_merge = ' '.join(txt_list)

word_str = get_word_str(txt_merge)

stop_words = ['自治体', '愛知', '名古屋', '県', '市', '区', '都道府県', '市町村', 'こと', 'もの', '等', '化', 'よう', 'ため', 'の', 'ところ']

wc = WordCloud(font_path=FONT_PATH,
    stopwords = set(stop_words),
    max_font_size = 80, 
    background_color="white",
    collocations = False,
    width=400,height=300).generate(word_str)

wc.to_file("./output/wordcloud_sample_str.png")

# co-occurrence network
## obama
# stopwords_txt = txt.get_stopword(top_n=2, min_freq=0)


# 形態素解析を行って、dfの列に名詞だけの列を1つ作る→ nlpotをすすめるのが良さそう。
def janome_text(text):
    from janome.tokenizer import Tokenizer
    import re
    
    #MeCabのインスタンスを作成（辞書はmecab-ipadic-neologdを使用）
    # mecab = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
    t = Tokenizer()
    token = t.tokenize(text)
    word_list = []

    for line in token:
        tmp = re.split('\t|,', str(line))
        # 名詞のみ対象
        if tmp[1] in ["名詞"]:
            word_list.append(tmp[0])
        # elif tmp[1] in ["形容詞"]:
        #     word_list.append(tmp[0])
            
    return word_list


#形態素結果をリスト化し、データフレームdfに結果を列追加する
df['words'] = df['comment'].apply(janome_text)

#表示
print(df.head())

txt_npt = nlplot.NLPlot(df, target_col='words')
txt_npt.build_graph(stopwords = stop_words, min_edge_frequency=30)
txt_npt.co_network(
    title='Co-occurrence network',
)


# correspondence analysis
# US大統領の就任スピーチをコレスポンデンス分析する
