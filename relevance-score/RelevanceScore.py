import numpy as np
import pandas as pd
import functools
from nltk import PorterStemmer


# 仅保存字段abstract中包含covid、-cov-2、cov2的数据
def search_focus(df):
    dfa = df[df['abstract'].str.contains('covid')]
    dfb = df[df['abstract'].str.contains('-cov-2')]
    dfc = df[df['abstract'].str.contains('cov2')]
    dfd = df[df['abstract'].str.contains('ncov')]
    frames=[dfa,dfb,dfc,dfd]
    df = pd.concat(frames)
    df=df.drop_duplicates(subset='title', keep="first")
    return df


# function to stem keyword list into a common base word
# 将关键字列表提取词干成一个通用的基本词
def stem_words(words):
    stemmer = PorterStemmer()
    singles = []
    for w in words:
        singles.append(stemmer.stem(w))
    return singles


def search_dataframe(df, search_words):
    # 如果一篇论文中包含我们要搜索的词，那么保留它
    df1 = df[functools.reduce(lambda a, b: a & b, (df['abstract'].str.contains(s) for s in search_words))]
    return df1


# function analyze search results for relevance with word count / abstract length
def search_relevance(rel_df, search_words):
    rel_df['score'] = ""
    # rel_df.loc['score']= ""
    for index, row in rel_df.iterrows():
        abstract = row['abstract']
        result = abstract.split()
        len_abstract = len(result)
        score = 0
        for word in search_words:
            score = score + result.count(word)
        final_score = (score / len_abstract)
        rel_score = score * final_score
        rel_df.loc[index, 'score'] = rel_score
    rel_df = rel_df.sort_values(by=['score'], ascending=False)
    # rel_df= rel_df[rel_df['score'] > .01]
    return rel_df


# function to get best sentences from the search results
def get_sentences(df1, search_words):
    df_table = pd.DataFrame(columns=["pub_date", "authors", "title", "excerpt", "rel_score"])
    for index, row in df1.iterrows():
        pub_sentence = ''
        sentences_used = 0
        # break apart the absracrt to sentence level
        sentences = row['abstract'].split('. ')
        # loop through the sentences of the abstract
        highligts = []
        for sentence in sentences:
            # missing lets the system know if all the words are in the sentence
            missing = 0
            # loop through the words of sentence
            for word in search_words:
                # if keyword missing change missing variable
                if word not in sentence:
                    missing = 1
                # if '%' in sentence:
                # missing=missing-1
            # after all sentences processed show the sentences not missing keywords
            if missing == 0 and len(sentence) < 1000 and sentence != '':
                sentence = sentence.capitalize()
                if sentence[len(sentence) - 1] != '.':
                    sentence = sentence + '.'
                pub_sentence = pub_sentence + '<br><br>' + sentence
        if pub_sentence != '':
            sentence = pub_sentence
            sentences_used = sentences_used + 1
            authors = row["authors"].split(" ")
            link = row['doi']
            title = row["title"]
            score = row["score"]
            linka = 'https://doi.org/' + link
            linkb = title
            sentence = '<p fontsize=tiny" align="left">' + sentence + '</p>'
            final_link = '<p align="left"><a href="{}">{}</a></p>'.format(linka, linkb)
            to_append = [row['publish_time'], authors[0] + ' et al.', final_link, sentence, score]
            df_length = len(df_table)
            df_table.loc[df_length] = to_append
    return df_table


################ main program ########################


# 从CSV文件中加载元数据
df=pd.read_csv('../input/oldversion/metadata.csv', usecols=['title','journal','abstract','authors','doi','publish_time','sha','full_text_file'])
df=df.fillna('no data provided')
df = df.drop_duplicates(subset='title', keep="first")
df=df[df['publish_time'].str.contains('2020')]
df["abstract"] = df["abstract"].str.lower()+df["title"].str.lower()
df=search_focus(df)
# list of lists of search terms
questions = [
    ['Q: What is the range of incubation periods for the disease in humans?'],
    ['Q: How long are individuals are contagious?'],
    ['Q: How long are individuals are contagious, even after recovery.'],
    ['Q: Does the range of incubation period vary across age groups?'],
    ['Q: Does the range of incubation period vary with children?'],
    ['Q: Does the range of incubation period vary based on underlying health?'],
    ['Q: What do we know about the basic reproduction number?'],
    ['Q: What is the prevalance of asymptomatic transmission?'],
    ['Q: What do we know about asymptomatic transmission in children?'],
    ['Q: What do we know about seasonality of transmission?'],
    ['Q: Informing decontamination based on physical science of the coronavirus?'],
    ['Q: What do we know about stability of the virus in environmental conditions?'],
    [
        'Q: What do we know about persistence of the virus on various substrates? (e.g., nasal discharge, sputum, urine, fecal matter, blood)'],
    [
        'Q: What do we know about persistence of the virus on various surfaces? (e,g., copper, stainless steel, plastic) '],
    ['Q: What do we know about viral shedding duration?'],
    ['Q: What do we know about viral shedding in fecal/stool?'],
    ['Q: What do we know about viral shedding from nasopharynx?'],
    ['Q: What do we know about viral shedding in blood?'],
    ['Q: What do we know about viral shedding in urine?'],
    ['Q: What do we know about implemtation of diagnostics?'],
    ['Q: What do we know about disease models?'],
    ['Q: What do we know about models for disease infection?'],
    ['Q: What do we know about models for disease transmission?'],
    ['Q: Are there studies about phenotypic change?'],
    ['Q: What is know about adaptations (mutations) of the virus?'],
    ['Q: What do we know about the human immune response and immunity?'],
    ['Q: Is population movement control effective in stopping transmission (spread)?'],
    ['Q: Effectiveness of personal protective equipment (PPE)?'],
    ['Q: What is the role of environment in transmission?']
]
# search=[['incubation','period','range','mean','%']]
search = [['incubation', 'period', 'range'],
          ['viral', 'shedding', 'duration'],
          ['asymptomatic', 'shedding'],
          ['incubation', 'period', 'age', 'statistically', 'significant'],
          ['incubation', 'period', 'child'],
          ['incubation', 'groups', 'risk'],
          ['basic', 'reproduction', 'number', '%'],
          ['asymptomatic', 'infection', '%'],
          ['asymptomatic', 'children'],
          ['seasonal', 'transmission'],
          ['contaminat', 'object'],
          ['environmental', 'conditions'],
          ['sputum', 'stool', 'blood', 'urine'],
          ['persistence', 'surfaces'],
          ['duration', 'viral', 'shedding'],
          ['shedding', 'stool'],
          ['shedding', 'nasopharynx'],
          ['shedding', 'blood'],
          ['shedding', 'urine'],
          ['diagnostics', 'point'],
          ['model', 'disease'],
          ['model', 'infection'],
          ['model', 'transmission'],
          ['phenotypic'],
          ['mutation'],
          ['immune', 'response'],
          ['restriction', 'movement'],
          ['protective', 'clothing'],
          ['transmission', 'routes']
          ]
q = 0
for search_words in search:
    search_words = stem_words(search_words)

    # 从过滤之后的数据中过滤那些abstract中不包含search_words的论文
    df1 = search_dataframe(df, search_words)

    # analyze search results for relevance
    df1 = search_relevance(df1, search_words)

    # get best sentences
    df_table = get_sentences(df1, search_words)

    length = df_table.shape[0]
    # limit 3 results
    df_table = df_table.head(15)
    df_table = df_table.drop(['rel_score'], axis=1)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df_table['excerpt'])

    if length < 1:
        print("No reliable answer could be located in the literature")
    q = q + 1
print('done')