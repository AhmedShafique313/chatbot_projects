import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
nltk.download('wordnet')

f = open('chatbot_data.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()
sent_tokens = nltk.sent_tokenize(raw)
word_tokens = nltk.word_tokenize(raw)

lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]

remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad you are talking to me"]

def greeting_response(text):
    for word in text.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    score = flat[-2]
    
    if score == 0:
        robo_response = "I apologize, but I don't understand."
    else:
        robo_response = sent_tokens[idx]
    
    sent_tokens.remove(user_response)
    return robo_response


flag = True
print("ChatBot: My name is ChatBot. I will answer your queries. If you want to exit, type 'bye'.")
while flag:
    user_response = input()
    user_response = user_response.lower()
    
    if user_response != 'bye':
        if user_response == 'thanks' or user_response == 'thank you':
            flag = False
            print("ChatBot: You're welcome!")
        else:
            if greeting_response(user_response) is not None:
                print("ChatBot: " + greeting_response(user_response))
            else:
                print("ChatBot: " + response(user_response))
    else:
        flag = False
        print("ChatBot: Goodbye! Have a great day.")


