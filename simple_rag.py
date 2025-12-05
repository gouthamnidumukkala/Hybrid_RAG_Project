import streamlit as st
import ollama

# Configure Streamlit page
st.set_page_config(
    page_title="Simple Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stop-button {
        background-color: #ff4b4b !important;
        color: white !important;
        border-radius: 20px !important;
        border: none !important;
        padding: 0.25rem 0.5rem !important;
        font-size: 0.8rem !important;
    }
    .generating-indicator {
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    .stButton > button[data-testid="baseButton-primary"] {
        background-color: #ff4b4b;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for chat history and model selection
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "qwen2.5:1.5b-instruct"
if "is_generating" not in st.session_state:
    st.session_state.is_generating = False
if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False

# Available models configuration
AVAILABLE_MODELS = {
    "qwen2.5:1.5b-instruct": "Qwen 2.5 1.5B (Instruct)",
    "llama3.2:1b-text-q2_K": "Llama 3.2 1B (Text Q2_K)",
    "qwen2.5-coder:1.5b-base-q4_K_M": "Qwen 2.5 Coder 1.5B (Base Q4_K_M)"
}

def get_ollama_response_stream(user_prompt, model_id):
    """Get streaming response from Ollama model"""
    try:
        stream = ollama.chat(
            model=model_id,
            messages=[{"role": "user", "content": user_prompt}],
            stream=True
        )
        
        full_response = ""
        for chunk in stream:
            if st.session_state.stop_generation:
                st.session_state.stop_generation = False
                return full_response + " [Response stopped by user]"
            
            if 'message' in chunk and 'content' in chunk['message']:
                full_response += chunk['message']['content']
                yield chunk['message']['content']
        
        return full_response
    except (ConnectionError, TimeoutError, KeyError) as e:
        return f"Error connecting to Ollama: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

def get_ollama_response(user_prompt, model_id):
    """Get response from Ollama model (non-streaming for testing)"""
    try:
        chat_response = ollama.chat(
            model=model_id,
            messages=[{"role": "user", "content": user_prompt}],
            stream=False
        )
        return chat_response['message']['content']
    except (ConnectionError, TimeoutError, KeyError) as e:
        return f"Error connecting to Ollama: {str(e)}"
    except Exception as e:
        return f"Unexpected error: {str(e)}"

# Main UI
st.title("🤖 Simple Chatbot")

# Model selection dropdown
col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"*Powered by {AVAILABLE_MODELS[st.session_state.selected_model]}*")
with col2:
    selected_model = st.selectbox(
        "Model:",
        options=list(AVAILABLE_MODELS.keys()),
        format_func=lambda x: AVAILABLE_MODELS[x],
        index=list(AVAILABLE_MODELS.keys()).index(st.session_state.selected_model),
        key="model_selector"
    )
    
    # Update session state if model changed
    if selected_model != st.session_state.selected_model:
        st.session_state.selected_model = selected_model
        st.rerun()

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to chat about?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response with streaming
    with st.chat_message("assistant"):
        # Create columns for response and stop button
        response_col, button_col = st.columns([6, 1])
        
        with button_col:
            # Stop button (only show during generation)
            stop_placeholder = st.empty()
        
        with response_col:
            response_placeholder = st.empty()
            
        # Show stop button and start generation
        st.session_state.is_generating = True
        st.session_state.stop_generation = False
        
        with stop_placeholder:
            if st.button("🛑 Stop", key=f"stop_{len(st.session_state.messages)}", help="Stop generating response"):
                st.session_state.stop_generation = True
        
        try:
            # Stream the response
            full_response = ""
            response_generator = get_ollama_response_stream(prompt, st.session_state.selected_model)
            
            for chunk in response_generator:
                if isinstance(chunk, str):
                    if chunk.startswith("Error"):
                        full_response = chunk
                        break
                    full_response += chunk
                    response_placeholder.markdown(full_response + "▌")  # Cursor effect
                    
                    # Check if stop was requested
                    if st.session_state.stop_generation:
                        break
            
            # Final response without cursor
            response_placeholder.markdown(full_response)
            
        except Exception as e:
            full_response = f"Error: {str(e)}"
            response_placeholder.markdown(full_response)
        
        finally:
            st.session_state.is_generating = False
            stop_placeholder.empty()  # Remove stop button
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar with options
with st.sidebar:
    st.header("Chat Options")
    
    # Generation status indicator
    if st.session_state.is_generating:
        st.success("🔄 Generating response...")
    else:
        st.info("✅ Ready for new message")
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()
    
    # Emergency stop for any ongoing generation
    if st.session_state.is_generating:
        if st.button("🛑 Emergency Stop", type="primary"):
            st.session_state.stop_generation = True
    
    st.markdown("---")
    st.markdown("### Model Info")
    st.info(f"Current: {AVAILABLE_MODELS[st.session_state.selected_model]}")
    
    # Test current model connection
    if st.button("Test Current Model"):
        try:
            test_response = ollama.chat(
                model=st.session_state.selected_model,
                messages=[{"role": "user", "content": "Hello!"}],
                stream=False
            )
            st.success("✅ Current model is working!")
        except (ConnectionError, TimeoutError, KeyError) as e:
            st.error(f"❌ Connection failed: {str(e)}")
            st.error("Make sure Ollama is running and the model is installed.")
        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")
    
    # Test all models
    if st.button("Test All Models"):
        for model_key, model_name in AVAILABLE_MODELS.items():
            try:
                test_response = ollama.chat(
                    model=model_key,
                    messages=[{"role": "user", "content": "Hello!"}],
                    stream=False
                )
                st.success(f"✅ {model_name} is working!")
            except (ConnectionError, TimeoutError, KeyError) as e:
                st.error(f"❌ {model_name} failed: {str(e)}")
            except Exception as e:
                st.error(f"❌ {model_name} error: {str(e)}")
