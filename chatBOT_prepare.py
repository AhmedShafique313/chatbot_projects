import nltk
import random
import string
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')

# Disable warnings
warnings.filterwarnings('ignore')

# Open and read the chatbot data file
file = open('D:\Programming languages programs\ChatBOT\chatbot_data.txt', 'r', errors='ignore')
readonConsole = file.read()
readonConsole = readonConsole.lower()

# Tokenize the data
sentence_tokens = nltk.sent_tokenize(readonConsole)
word_tokens = nltk.word_tokenize(readonConsole)

SentenceToken = sentence_tokens[:4]
print(SentenceToken)
WordToken = word_tokens[:4]
print(WordToken)

print("\n \n \n \n          ChatBOT loading...  \n \n \n ")

# Preprocessing
Lemmer = nltk.stem.WordNetLemmatizer()

def LemTokens(tokens):
    return [Lemmer.lemmatize(token) for token in tokens]

Remove_Punctuation_from_Dictionary = dict((ord(punctuation), None) for punctuation in string.punctuation)

def LemNormalize(text):
    return LemTokens(nltk.word_tokenize(text.lower().translate(Remove_Punctuation_from_Dictionary)))

# Greetings function
Greeting_Input = ("hi", "hello", "whats UP", "hey", "hi there", "hye", "salam")
Greeting_Response = ("hi", "hey", "hello", "Hi", "salam", "hi there", "whats up")

def Greetings(scentences):
    for word in scentences.split():
        if word.lower() in Greeting_Input:
            return random.choice(Greeting_Response)

# Vectorization and response generation
def Response(User_Response):
    ChatBOT_Response = " "
    sentence_tokens.append(User_Response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    Tfidf = TfidfVec.fit_transform(sentence_tokens)
    vals = cosine_similarity(Tfidf[-1], Tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfdif = flat[-2]
    
    if req_tfdif == 0:
        ChatBOT_Response = ChatBOT_Response + "I am sorry, I don't understand"
        return ChatBOT_Response
    else:
        ChatBOT_Response = ChatBOT_Response + sentence_tokens[idx]
        return ChatBOT_Response

# Main loop
if __name__ == "__main__":
    flag = True
    print("This is Udemy ChatBOT for answer your questions... ")
    while flag:
        User_Response = input()
        User_Response = User_Response.lower()
        if User_Response != "bye" or User_Response!= "Bye":
            if User_Response == "thanks" or User_Response == "thank you" or User_Response == "thankyou":
                flag = False
                print("You're welcome!")
            else:
                if Greetings(User_Response) is not None:
                    print("Chatbot: " + Greetings(User_Response))
                else:
                    print("Chatbot:", end=' ')
                    print(Response(User_Response))
                    sentence_tokens.remove(User_Response)
        else:
            flag = False
            print("Chatbot: Goodbye!")
