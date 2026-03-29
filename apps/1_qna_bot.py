from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

st.title("🤔 AnswerBuddy AI ,Ask Any Qna")
st.markdown("My QnA Bot with LangChain and Google Gemini !")


# messages is like a key in a session state that will store the conversation
if 'messages' not in st.session_state:
    #if there is no messages key then create one
    st.session_state.messages = []

for message in st.session_state.messages:
    role = message['role']
    content = message['content']
    st.chat_message(role).markdown(content) 

query = st.chat_input("Ask me anything..")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    st.chat_message('user').markdown(query)
    res = llm.invoke(query)
    st.chat_message('ai').markdown(res.content)
    st.session_state.messages.append({"role": "ai", "content": res.content})


# role tells use who's message it is user or ai