import streamlit as st
import pickle
import re
import nltk
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
nltk.download('stopwords')
nltk.download('punkt')
clf=pickle.load(open("clf.pkl","rb"))
tfidf=pickle.load(open("tfidf.pkl","rb"))
def cleanResume(txt):
    cleanTxt=re.sub('http\S+\s',' ',txt)
    cleanTxt=re.sub('RT|CC',' ',cleanTxt)
    cleanTxt=re.sub('#\S+\s',' ',cleanTxt)
    cleanTxt=re.sub('@\S+',' ',cleanTxt)
    cleanTxt=re.sub('[%s]'%re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""),' ',cleanTxt)
    cleanTxt=re.sub(r'[^\x00-\x7f]',' ',cleanTxt)
    cleanTxt=re.sub('\s+',' ',cleanTxt)
    return cleanTxt
def main():
    st.title("Resume Detection")
    upload_file = st.file_uploader("Upload your resume", type=["txt", "pdf"]) 
    if upload_file is not None:
        try:
            resume_bytes = upload_file.read()
            resume_text = resume_bytes.decode('utf-8')
        except UnicodeDecodeError:
            resume_text = resume_bytes.decode('latin-1')
        clean_resume = cleanResume(resume_text)
        y = tfidf.transform([clean_resume])
        prediction_id = clf.predict(y)[0]
        st.write(prediction_id)
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
if __name__=="__main__":
    main()