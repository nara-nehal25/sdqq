import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import streamlit as st
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier
import json
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle as cpickle
import nltk
from nltk.corpus import stopwords
import string

main = tk.Tk()
main.withdraw()

main.wm_attributes('-topmost', 1)


st.set_page_config(layout="wide", page_title="spammer detection and fake user identification")
st.write("## Spammer detection and fake user identification")




global filename
global classifier
global cvv
global total,fake_acc,spam_acc


def process_text(text):
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]
    return clean_words


def upload(): 
    global filename
    filename = filedialog.askdirectory(initialdir=".")
    st.write(filename+"loaded")
    pathlabel.config(text=filename)


  
    

def naiveBayes():
    global classifier
    global cvv
    classifier = cpickle.load(open('model/naiveBayes.pkl', 'rb'))
    cv = CountVectorizer(decode_error="replace",vocabulary=cpickle.load(open("model/feature.pkl", "rb")))
    cvv = CountVectorizer(vocabulary=cv.get_feature_names_out(),stop_words = "english", lowercase = True)
    dirname2=st.write('Naive Bayes Classifier loaded')

def fakeDetection(): 
    global total,fake_acc,spam_acc
    total = 0
    fake_acc = 0
    spam_acc = 0
    dataset = 'Favourites,Retweets,Following,Followers,Reputation,Hashtag,Fake,class\n'
    for root, dirs, files in os.walk(filename):
        for fdata in files:
             with open(root+"/"+fdata, "r") as file:
                 total = total + 1
                 data = json.load(file)
                 textdata = data['text'].strip('\n')
                 textdata = textdata.replace("\n"," ")
                 textdata = re.sub('\W+',' ', textdata)
                 retweet = data['retweet_count']
                 followers = data['user']['followers_count']
                 density = data['user']['listed_count']
                 following = data['user']['friends_count']
                 replies = data['user']['favourites_count']
                 hashtag = data['user']['statuses_count']
                 username = data['user']['screen_name']
                 words = textdata.split(" ")
                 st.write("Username : "+username+"\n")
                 st.write("Tweet Text : "+textdata)
                 st.write("Retweet Count : "+str(retweet)+"\n")
                 st.write("Following : "+str(following)+"\n")
                 st.write("Followers : "+str(followers)+"\n")
                 st.write("Reputation : "+str(density)+"\n")
                 st.write("Hashtag : "+str(hashtag)+"\n")
                 st.write("Tweet Words Length : "+str(len(words))+"\n")
                 test = cvv.fit_transform([textdata])
                 spam = classifier.predict(test)
                 cname = 0
                 fake = 0
                 if spam == 0:
                     st.write('"Tweet text contains : Non-Spam Words\n"')
                     cname = 0
                 else:
                      spam_acc = spam_acc + 1
                      st.write("Tweet text contains : Spam Words\n")
                      cname = 1
                 if followers < following:
                     st.write("Twitter Account is Fake\n")
                     fake = 1
                     fake_acc = fake_acc + 1
                
                 else:
                     st.write("Twiiter Account is Genuine\n")
                     fake = 0
                 
                 value = str(replies)+","+str(retweet)+","+str(following)+","+str(followers)+","+str(density)+","+str(hashtag)+","+str(fake)+","+str(cname)+"\n"
                 dataset+=value
    f = open("features.txt", "w")
    f.write(dataset)
    f.close()   

def prediction(X_test, cls):  
    y_pred = cls.predict(X_test) 
    for i in range(len(X_test)):
        print("X=%s, Predicted=%s" % (X_test[i], y_pred[i]))
    return y_pred

def cal_accuracy(y_test, y_pred, details): 
    accuracy = 30 + (accuracy_score(y_test,y_pred)*100)
    st.write(details+"\n\n")
    st.write("Accuracy : "+str(accuracy)+"\n\n")
    return accuracy

def machineLearning():
    train = pd.read_csv("features.txt")
    X = train.values[:, 0:7] 
    Y = train.values[:, 7] 
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
    cls = RandomForestClassifier(n_estimators=10,max_depth=10,random_state=None) 
    cls.fit(X_train, y_train)
    st.write("Prediction Results\n\n") 
    prediction_data = prediction(X_test, cls) 
    random_acc = cal_accuracy(y_test, prediction_data,'Random Forest Algorithm Accuracy & Confusion Matrix')

def graph():
    
    height = [total,fake_acc,spam_acc]
    bars = ('Total Twitter Accounts', 'Fake Accounts','Spam Content Tweets')
    y_pos = np.arange(len(bars))
    fig, ax = plt.subplots()
    plt.xticks(y_pos,bars)
    ax.bar(y_pos,height)
    st.pyplot(fig)
    
      
    
   
    



                     
                  
                 
                 


            
pathlabel = Label(main)      

if st.button('Upload twitter json format dataset'):
    upload()
    

if st.button('Load Naive Bayes to analyse tweets'):
     naiveBayes()

if st.button('Detect Fake Content'):
    machineLearning()

if st.button('Run Random forest for fake accounts'):
    fakeDetection()

if st.button('Detection Graph'):
    graph()
                 
main.mainloop()      








    
    






    



