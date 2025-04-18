import pickle
import pandas
import numpy as np
from h5py.h5pl import append
from keras.utils import to_categorical
from konlpy.tag import Okt
from keras.preproceessing.sequence import pad_seqeunces
from keras.models import load_model

from job04_preprocess import labeled_y, onehot_y, x_pad

df = pd.read_csv('./crawling_data/naver_headline_news_20250418.csv')
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)
print(df.head())
df.into()
print(df.category.value_counts())

X = df.titles
Y = df.category

with open('./models/encoder.pickle', 'rb') as f:
    # 여기에 파일을 열고 처리하는 코드 작성
    encoder = pickle.load(f)
label = encoder.classes_
print(label)

labeled_y = encoder.transform(Y)
onehot_y = to_categorical(labeled_y)
print(onehot_y)

okt = Okt()
for i in range(len(X)):
    x[i] = okt.morphs(X[i], stem=True)
print(X[:10])

with open('./models/token_max_25.pickle', 'rb') as f:
     token = pickle.load(f)
tokened_x = token.texts_to_sequences(X)
print(tokened_x[:5])

for i in range(len(tokened_x)):
    if len(tokened_x[i]) > 25:
         tokened_x[i] = tokened_x[i][:25]
x_pad = pad_seqeunces(tokened_x, 25)
print(x_pad)
model = load_model(news_section_classfication_model_0.7207527756690979.h5) as f:
preds = model.predict(x_pad)
print(preds)

predict_section = []
for pred in preds:
    most = (label[np.argmax(pred)])
    pred[np.argmax(pred)] = 0
    second = label[np.argmax(pred)]
    predict_section.append([append(most, second)])
print(predict_section)

df['predict'] = predict_section
print(df.head(30))

score = model.evaluate(x_pad, onehot_y)
print(score[1])

df['OX'] = 0
for i in range(len(df)):
     if df.loc[i, 'category'] in df.loc[i, 'predict']:
         df.loc[i, 'OX'] = 1
print(df.OX.mean())
