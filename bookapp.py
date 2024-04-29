import pickle
import streamlit as st 
import numpy as np 

model=pickle.load(open('saving/model.pkl','rb'))
books_name=pickle.load(open('saving/book_name.pkl','rb'))
final_data_frame=pickle.load(open('saving/final_data_frame.pkl','rb'))
book_pivot=pickle.load(open('saving/book_pivot.pkl','rb')) 


def fetch_poster(suggesitions):
     book_name=[]
     ids_index=[]
     poster_url=[]
     for book_id in suggesitions[0]:
          book_name.append(book_pivot.index[book_id])
     for  name in book_name :
          ids=np.where(final_data_frame['title']==name)[0][0]
          ids_index.append(ids)
     for idx in ids_index:
          url=final_data_frame.loc[idx,'image_url']
          poster_url.append(url)
     return poster_url        
          
             


     
def recommend(book_name):
    print("Selected book:", book_name)
    book_list=[]
    book_id=np.where(book_pivot.index==book_name)[0][0]
    
    distance,suggesitions=model.kneighbors(book_pivot.iloc[book_id,:].values.reshape(1,-1),n_neighbors=6)
    poster_url=fetch_poster(suggesitions)

    for books_id in suggesitions :
         
         for j in books_id :
              book_list.append(book_pivot.index[j])
    
    return book_list ,poster_url 




st.set_page_config(page_title="Book Recommender System",page_icon="📚")

st.header('Books Recommender System using Machine Learning')
     
selected_book=st.selectbox("Type or select Book name ",books_name)
     
             
if st.button('Show Recommendation'):
    recommendation_books, urls = recommend(selected_book)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommendation_books[1])
        st.image(urls[1])
    with col2:
        st.text(recommendation_books[2])
        st.image(urls[2])
    with col3:
        st.text(recommendation_books[3])
        st.image(urls[3])

    with col4:
        st.text(recommendation_books[4])
        st.image(urls[4])
    with col5:
        st.text(recommendation_books[5])
        st.image(urls[5])

