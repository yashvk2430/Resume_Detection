import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
df=pd.read_csv("UpdatedResumeDataSet.csv")
print(df.head())
print(df.shape)
print(df.info())
print(df["Category"].value_counts())
plt.figure(figsize=(10,5))
sns.countplot(x="Category",data=df)
plt.show()
counts=df["Category"].value_counts()
labels=df["Category"].unique()
plt.figure(figsize=(15,10))
plt.pie(counts,labels=labels,autopct='%1.1f%%',shadow=True,colors=plt.cm.coolwarm(np.linspace(0,1,3)))
plt.show() 
def cleanResume(txt):
    cleanTxt=re.sub('http\S+\s',' ',txt)
    cleanTxt=re.sub('RT|CC',' ',cleanTxt)
    cleanTxt=re.sub('#\S+\s',' ',cleanTxt)
    cleanTxt=re.sub('@\S+',' ',cleanTxt)
    cleanTxt=re.sub('[%s]'%re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',cleanTxt)
    cleanTxt=re.sub(r'[^\x00-\x7f]',' ',cleanTxt)
    cleanTxt=re.sub('\s+',' ',cleanTxt)
    return cleanTxt
df["Resume"]=df["Resume"].apply(lambda x:cleanResume(x))
print(df["Resume"].head())
print(df['Category'].unique())
le=LabelEncoder()
le.fit_transform(df["Category"])
df["Category"]=le.transform(df["Category"])
print(df["Category"].head())
print(df["Category"].unique())
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf=TfidfVectorizer(stop_words='english')
tfidf.fit(df['Resume'])
requried_text=tfidf.transform(df['Resume'])
print(requried_text)
X_train,X_test,y_train,y_test=train_test_split(requried_text,df["Category"],test_size=0.2,random_state=42)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
clf=OneVsRestClassifier(KNeighborsClassifier())
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test)
print(y_pred)
print(f"accuracy of training dataset: {accuracy_score(y_test,y_pred)}")
myresume="""I am a data scientist specializing in machine
learning, deep learning, and computer vision. With
a strong background in mathematics, statistics,
and programming, I am passionate about
uncovering hidden patterns and insights in data.
I have extensive experience in developing
predictive models, implementing deep learning
algorithms, and designing computer vision
systems. My technical skills include proficiency in
Python, Sklearn, TensorFlow, and PyTorch.
What sets me apart is my ability to effectively
communicate complex concepts to diverse
audiences. I excel in translating technical insights
into actionable recommendations that drive
informed decision-making.
If you're looking for a dedicated and versatile data
scientist to collaborate on impactful projects, I am
eager to contribute my expertise. Let's harness the
power of data together to unlock new possibilities
and shape a better future.
Contact & Sources
Email: 611noorsaeed@gmail.com
Phone: 03442826192
Github: https://github.com/611noorsaeed
Linkdin: https://www.linkedin.com/in/noor-saeed654a23263/
Blogs: https://medium.com/@611noorsaeed
Youtube: Artificial Intelligence
ABOUT ME
WORK EXPERIENCE
SKILLES
NOOR SAEED
LANGUAGES
English
Urdu
Hindi
I am a versatile data scientist with expertise in a wide
range of projects, including machine learning,
recommendation systems, deep learning, and computer
vision. Throughout my career, I have successfully
developed and deployed various machine learning models
to solve complex problems and drive data-driven
decision-making
Machine Learnine
Deep Learning
Computer Vision
Recommendation Systems
Data Visualization
Programming Languages (Python, SQL)
Data Preprocessing and Feature Engineering
Model Evaluation and Deployment
Statistical Analysis
Communication and Collaboration
"""
import pickle 
pickle.dump(tfidf,open("tfidf.pkl","wb"))
pickle.dump(clf,open("clf.pkl","wb"))
def pred(input_resume):
    # Preprocess the input text (e.g., cleaning, etc.)
    cleaned_text = cleanResume(input_resume) 

    # Vectorize the cleaned text using the same TF-IDF vectorizer used during training
    vectorized_text = tfidf.transform([cleaned_text])
    
    # Convert sparse matrix to dense
    vectorized_text = vectorized_text.toarray()

    # Prediction
    predicted_category = clf.predict(vectorized_text)

    # get name of predicted category
    predicted_category_name = le.inverse_transform(predicted_category)

    return predicted_category_name[0]

print(pred(myresume))