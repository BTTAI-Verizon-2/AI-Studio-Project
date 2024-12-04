#GUI
import streamlit as st
from main import datachat

chat_object = datachat('./training_and_test_data/mobile_subscriber_churn_train.xlsx')

conn = st.button('connect')
if conn:
    chat_object.vectorize()

st.title(":blue[Customer Churn Database]")
col1, col2, col3 = st.columns(3)
with col3:
    st.subheader("powered by GenAI")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == 'user':
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if message["role"] == 'assistant':
        with st.chat_message(message["role"]):
            st.dataframe(message["content"],hide_index=True)

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = chat_object.data_ops(prompt)
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        #st.markdown(response)
        st.dataframe(response,hide_index=True)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})