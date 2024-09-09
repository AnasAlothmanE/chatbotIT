import json
import string
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import joblib

# تحميل مكتبة nltk
nltk.download("punkt")
nltk.download("wordnet")
nltk.download("stopwords")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('arabic'))

# تحميل بيانات النية
with open('modified_data.json', 'r') as file:
    data = json.load(file)

# معالجة البيانات
words = []
classes = []
documents = []
all_patterns = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if
                  word not in stop_words and word not in string.punctuation]
        words.extend(tokens)
        documents.append((" ".join(tokens), intent["tag"]))
        all_patterns.append(" ".join(tokens))

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

words = sorted(set(words))
classes = sorted(set(classes))

# إعداد Tokenizer لتحويل الكلمات إلى تسلسل رقمي
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(all_patterns)

# تحويل النصوص إلى تسلسل رقمي
sequences = tokenizer.texts_to_sequences(all_patterns)
max_sequence_len = max([len(seq) for seq in sequences])
X = pad_sequences(sequences, maxlen=max_sequence_len, padding='post')

# تحويل الفئات إلى أرقام
label_binarizer = LabelBinarizer()
y = label_binarizer.fit_transform([doc[1] for doc in documents])

# تقسيم البيانات إلى مجموعة تدريب واختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# بناء نموذج بسيط للتصنيف باستخدام LSTM
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=100, input_length=max_sequence_len))
model.add(LSTM(128))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1, validation_data=(X_test, y_test))

# حفظ النموذج المدرب
model.save('text_classifier_model.keras')

# حفظ Tokenizer و LabelBinarizer
joblib.dump(tokenizer, 'tokenizer.pkl')
joblib.dump(label_binarizer, 'label_binarizer.pkl')
