import streamlit as st
import pickle
import string
import nltk
#nltk.download('wordnet')
#nltk.download('stopwords')
from nltk.corpus import stopwords 

def load_model():
    with open('yelp_nb_model.pkl', 'rb') as f:
        objects = pickle.load(f)
    return objects

def text_process(txt):
  no_punc=[chr for chr in txt if chr not in string.punctuation]
  out_txt=''.join(no_punc)
  out_list = [word for word in out_txt.split() if word.lower() not in stopwords.words('english') and word.lower().isalpha()]
  return " ".join(out_list)

def lemmatize_text(txt):
  word_tokenizer = nltk.tokenize.WhitespaceTokenizer()
  lemmatizer = nltk.stem.WordNetLemmatizer()
  out_list = [lemmatizer.lemmatize(word) for word in word_tokenizer.tokenize(txt) ]
  return " ".join(out_list)

def predict(model, input_data):
    prediction = model.predict(input_data)
    return prediction

def main():
    st.title("User Review Classification")

    try:
      objects = load_model()

      model = objects['model']
      cv = objects['cv']


      user_input = st.text_input("Enter a sample review text")
      
      x_new = text_process(user_input)
      x_new = lemmatize_text(x_new)
      x_new = cv.transform([x_new]).toarray()

      if st.button("Predict"):
        prediction = model.predict(x_new)
        result = ("Positive" if prediction[0]>0 else "Negative")  
        st.write(result)

    except Exception as e:
      st.error(f"Error: {e}")

if __name__ == "__main__":
    main()