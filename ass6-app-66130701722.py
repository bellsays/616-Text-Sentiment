import streamlit as st
import pickle
import re
from pythainlp.util import normalize
 
# โหลดโมเดล LogisticRegression
model_bay = pickle.load(open('naive_bayes_model.sav', 'rb'))
 
# โหลด TfidfVectorizer
tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.sav', 'rb'))
 
# ฟังก์ชันทำความสะอาดข้อความ
def TextClean(text):
  text = re.sub('<[^>]*>', '', text)
  emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
  text = (re.sub('[\W]+', ' ', text.lower()) + ' '.join(emoticons).replace('-', ''))
  text = normalize(text)
  return text
 
# ตั้งชื่อหัวข้อหน้าเว็บ
st.title("Text Sentiment Prediction Using Naive bayes Model Created By Supattra saekow ID: 6613070172")
 
def main():
  # สร้าง Sidebar สำหรับรับข้อความ
  text = st.text_input("ป้อนข้อความของคุณ")
 
  # ทำความสะอาดข้อความ
  text = TextClean(text)
 
  # แปลงข้อความให้เป็นเวกเตอร์ TF-IDF
  X_new_tfidf = tfidf_vectorizer.transform([text])
 
  # ทำนายผล
  prediction = model_bay.predict(X_new_tfidf)
 
  # แสดงผลลัพธ์
  if prediction == 'pos':
    st.success("Positive")
  else:
    st.error("Negative")
 
if __name__ == "__main__":
  main()
