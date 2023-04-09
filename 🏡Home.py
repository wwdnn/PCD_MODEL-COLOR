import streamlit as st

st.set_page_config(
  page_title="Home", 
  page_icon=":house:", 
  layout="centered", 
  initial_sidebar_state="expanded"
)


st.title("Hallo Semuanya!✨")
st.text("Izin memperkenalkan diri terlebih dahulu yaa!!")

col1, col2 = st.columns([1,2])
with col1:
  st.image("./Tugas/Meeting 4/Images/foto-profil.jpg", width=200)
with col2:
  st.markdown("<h1 style='text-align: center; color: white;'>Wildan Setya Nugraha</h1>", unsafe_allow_html=True)
  st.markdown("<h2 style='text-align: center; color: white;'>211511032</h2>", unsafe_allow_html=True)
  st.markdown("<h2 style='text-align: center; color: white;'>Politeknik Negeri Bandung</h2>", unsafe_allow_html=True)


st.balloons()