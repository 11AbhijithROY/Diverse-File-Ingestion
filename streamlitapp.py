import streamlit as st
from multi_agent_module import main
st.set_page_config(page_title="Multi Agent Chat")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 


st.title("Multi Agent Interface")

user_input = st.text_input("Ask a question...", placeholder="What's the weather like today?", label_visibility='collapsed')
# st.write(user_input)

if user_input:
    gc_manager, user_proxy = main()
    chat_result = user_proxy.initiate_chat(recipient=gc_manager, message=user_input)
    st.session_state.chat_history.append(("User", user_input))
    st.session_state.chat_history.append(("AI", chat_result))

    # st.text_area("Answer: ", value=chat_result.chat_history, height=200)
    for message in chat_result.chat_history:
        if message.get("role") in ["tool", "user"]:
            if "content" in message:
                correct_answer = message["content"]
                st.write(correct_answer)
                break
    st.rerun()

    st.markdown("### Chat History")
    for msg in st.session_state.chat_history:
        st.write(f"**{msg['role']}:** {msg['content']}")

    
    # st.write(chat_result)