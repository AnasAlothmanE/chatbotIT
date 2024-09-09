import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import flet as ft
from deep_translator import GoogleTranslator

lemmatizer = WordNetLemmatizer()

# تحميل النموذج المدرب
model = load_model('chatbot_model.h5')
with open('modified_data.json', 'r') as file:
    data_int = json.load(file)
# تحميل الكلمات والتصنيفات
with open('words_classes.json', 'r') as file:
    data = json.load(file)
    words = data['words']
    classes = data['classes']

# معالجة النص
def clean_text(text):
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens if word not in string.punctuation]
    return tokens

def bag_of_words(text):
    tokens = clean_text(text)
    bag = [0] * len(words)
    for w in tokens:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(text):
    bow = bag_of_words(text)
    result = model.predict(np.array([bow]))[0]
    threshold = 0.2
    predictions = [[i, r] for i, r in enumerate(result) if r > threshold]
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions

def get_response(intents):
    if len(intents) > 0:
        for intent in data_int["intents"]:
            if intent["tag"] == classes[intents[0][0]]:
                return random.choice(intent["responses"])
    else:
        return "I'm sorry, I didn't understand your question. Could you please rephrase it?"

def detect_language(text):
    if any("\u0600" <= char <= "\u06FF" for char in text):  # Check if Arabic script is present
        return 'ar'
    return 'en'

# إنشاء واجهة المستخدم باستخدام flet
def main(page: ft.Page):
    page.title = "Chatbot Interface"
    page.vertical_alignment = ft.MainAxisAlignment.CENTER

    chat_box = ft.Column(
        controls=[],
        spacing=10,
        scroll="auto",  # السماح بالتمرير
        height=400  # تحديد ارتفاع العمود لكي يعمل التمرير
    )

    user_input = ft.TextField(
        hint_text="Type your message here...",
        on_submit=lambda e: handle_user_input(e, chat_box, user_input)
    )

    send_button = ft.IconButton(
        icon=ft.icons.SEND,
        on_click=lambda e: handle_user_input(e, chat_box, user_input)
    )

    page.add(
        ft.Row(
            controls=[
                user_input,
                send_button
            ],
            alignment=ft.MainAxisAlignment.CENTER,
        ),
        chat_box
    )

def handle_user_input(e, chat_box, user_input):
    user_message = user_input.value
    if user_message:
        chat_box.controls.append(ft.Text(f"You: {user_message}"))

        # تحديد لغة المدخلات
        lang = detect_language(user_message)

        # إذا كانت اللغة العربية، قم بالترجمة إلى الإنجليزية
        if lang == 'ar':
            user_message = GoogleTranslator(source='ar', target='en').translate(user_message)

        intents = predict_class(user_message)
        response = get_response(intents)

        # إذا كانت المدخلات بالعربية، قم بترجمة الرد إلى العربية
        if lang == 'ar':
            response = GoogleTranslator(source='en', target='ar').translate(response)

        chat_box.controls.append(ft.Text(f"Bot: {response}"))
        user_input.value = ""
        e.page.update()

# تشغيل التطبيق
ft.app(target=main, view=ft.AppView.WEB_BROWSER)
