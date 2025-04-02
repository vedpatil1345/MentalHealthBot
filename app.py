from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import streamlit as st
num=1
# Initialize the ChatGroq model
llm = ChatGroq(
    model="llama-3.2-90b-text-preview",
    groq_api_key="api key",  # Ensure the API key is valid
    temperature=0
)

# Set up the prompt templates
response_template = PromptTemplate.from_template(
    """
    ### Instructions:
    You are a compassionate AI assistant dedicated to supporting mental well-being. Based on the conversation history, craft a thoughtful response that addresses the user's input and provides support.
    
    ### Conversation History:
    {formatted_history}

    ### User's Input:
    User: {user_input}

    ### Your Response(with emojis):
    Assistant:
    """
)

question_template = PromptTemplate.from_template(
    """
    ### Instructions:
    You are a compassionate AI assistant dedicated to supporting mental well-being. Based on the conversation history, craft a thoughtful, open-ended question that gently encourages the user to share their feelings and reflect on their thoughts.

    ### Conversation History:
    {formatted_history}

    ### Your Question:
    Question:
    give question only 
    """
)

# Streamlit app title
st.title("Mental Health Chatbot")

# Initialize session state variables
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = "How are you feeling today?"
if 'process_input' not in st.session_state:
    st.session_state.process_input = False

# Display conversation history
st.subheader("Conversation History")

for q, u, a in st.session_state.conversation_history:
    st.success(f"{num}. Q: {q}")
    st.success(f"You: {u}")
    st.success(f"Bot: {a}")
    st.text("")  # Add a blank line for separation
    num=num+1
# User input
user_input = st.text_input(st.session_state.current_question, key="user_input")

# Button to submit the input
if st.button("Submit"):
    st.session_state.process_input = True

# Process input when the flag is set
if st.session_state.process_input and user_input:
    # Format conversation history for prompt
    formatted_history = "\n".join(
        f"Question: {msg[0]}\nUser: {msg[1]}\nAssistant: {msg[2]}" for msg in st.session_state.conversation_history
    )

    # Prepare the prompt for the AI
    full_prompt = response_template.format(formatted_history=formatted_history, user_input=user_input)

    # Generate a response with a loading spinner
    with st.spinner('Generating response...'):
        try:
            response = llm.predict(full_prompt)
            st.success(f"Bot: {response}")
            
            # Add the current interaction to the conversation history
            st.session_state.conversation_history.append((st.session_state.current_question, user_input, response))
            
            # Generate the next question
            question_prompt = question_template.format(formatted_history=formatted_history)
            next_question = llm.predict(question_template.format(formatted_history=formatted_history))
            st.session_state.current_question = next_question.strip()
            
            # Reset the process_input flag
            st.session_state.process_input = False
            
            # Clear the input box
            st.user_input = ""
            
            # Rerun the script to update the UI
            st.rerun()

        except Exception as e:
            st.error(f"Error: {e}")
            st.session_state.process_input = False

# Reset process_input flag if there's no user input
if not user_input:
    st.session_state.process_input = False
