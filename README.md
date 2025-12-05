# 🤖 Mahi Hybrid RAG Assistant

A professional-grade Retrieval Augmented Generation (RAG) system built with Streamlit, Ollama, and modern AI technologies.

## ✨ Features

- **💬 Chat Mode**: Direct conversation with specialized AI assistants
- **📚 RAG Mode**: AI with document knowledge and context retrieval
- **🔧 Specialized Chatbots**: Programmer, Teacher, Creative Writer, Analyst, and more
- **📄 Document Processing**: Support for PDF, DOCX, TXT, and Markdown files
- **🔍 Vector Search**: Advanced similarity search with ChromaDB
- **⚙️ Configurable**: Extensive configuration options
- **💾 Conversation Management**: Save and load chat history
- **🎨 Professional UI**: Clean, responsive Streamlit interface

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI/LLM**: Ollama (Llama 3.2)
- **Vector Database**: ChromaDB
- **Embeddings**: Sentence Transformers
- **Document Processing**: PyPDF2, python-docx
- **Configuration**: YAML + Environment Variables

## 📁 Project Structure

```
Mahi_Hybrid_RAG/
├── src/                          # Source code
│   ├── core/                     # Core functionality
│   │   ├── chatbot.py           # Chatbot classes
│   │   └── rag.py               # RAG system
│   ├── ui/                      # User interface
│   │   └── streamlit_app.py     # Main Streamlit app
│   └── utils/                   # Utilities
│       └── config.py            # Configuration management
├── config/                      # Configuration files
│   ├── config.yaml              # Main configuration
│   └── .env.example             # Environment template
├── data/                        # Data directory
│   ├── documents/               # Uploaded documents
│   └── models/                  # Model files
├── logs/                        # Application logs
├── requirements.txt             # Python dependencies
├── setup.sh                     # Setup script
├── run_app.sh                   # Run script
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Setup

```bash
# Clone or ensure you're in the project directory
cd Mahi_Hybrid_RAG

# Run the setup script
./setup.sh
```

This will:
- Create a virtual environment
- Install all dependencies
- Set up Ollama and download the default model
- Create necessary directories
- Set up configuration files

### 2. Configuration (Optional)

```bash
# Copy and edit environment variables
cp config/.env.example config/.env
# Edit config/.env with your API keys (if needed)

# Modify config/config.yaml for custom settings
```

### 3. Run the Application

```bash
# Start the application
./run_app.sh
```

The application will be available at `http://localhost:8501`

## 📋 Manual Installation

If you prefer manual installation:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir -p data/{documents,models} logs

# Install and start Ollama
# Visit https://ollama.ai/ for installation instructions
ollama pull llama3.2:1b

# Start the application
streamlit run src/ui/streamlit_app.py
```

## 🔧 Configuration

### Main Configuration (`config/config.yaml`)

- **Models**: Configure available AI models
- **RAG Settings**: Chunk size, overlap, similarity threshold
- **UI Settings**: Page title, layout, theme
- **Chat Settings**: History limit, streaming, temperature

### Environment Variables (`config/.env`)

- `OLLAMA_API_KEY`: Ollama API key (if required)
- `HUGGINGFACE_API_KEY`: Hugging Face API key (for embeddings)
- `OPENAI_API_KEY`: OpenAI API key (alternative embeddings)

## 🎯 Usage

### Chat Mode

1. Select "Chat Mode" in the sidebar
2. Choose a model and specialty (Programmer, Teacher, etc.)
3. Start chatting!

### RAG Mode

1. Select "RAG Mode" in the sidebar
2. Upload documents using the document management section
3. Ask questions about your documents
4. View sources and relevance scores

### Features

- **Model Selection**: Switch between different Llama models
- **Temperature Control**: Adjust response creativity
- **Conversation History**: Automatic chat history
- **Save/Load**: Export and import conversations
- **Document Management**: Upload, process, and delete documents
- **Source Tracking**: See which documents influenced responses

## 📚 Supported Document Formats

- **PDF** (`.pdf`): Adobe PDF documents
- **Word** (`.docx`): Microsoft Word documents
- **Text** (`.txt`): Plain text files
- **Markdown** (`.md`): Markdown documents

## 🔍 Advanced Features

### Specialized Chatbots

- **Programmer**: Expert in coding and software development
- **Teacher**: Educational explanations and learning support
- **Creative**: Creative writing and artistic content
- **Analyst**: Data analysis and research insights
- **Customer Support**: Professional customer service
- **Medical**: Health and medical information
- **Legal**: Legal information and guidance

### RAG System

- **Intelligent Chunking**: Smart document segmentation
- **Vector Similarity**: Semantic search using embeddings
- **Context Ranking**: Relevance-based document retrieval
- **Source Attribution**: Track information sources
- **Multi-document**: Query across multiple documents

## 🛡️ Error Handling

The application includes comprehensive error handling:

- **Model Loading**: Graceful fallbacks for missing models
- **Document Processing**: Error reporting for failed uploads
- **API Failures**: Informative error messages
- **Configuration**: Default values for missing settings

## 📊 Logging

Logs are stored in the `logs/` directory:

- `streamlit_app.log`: Main application logs
- `conversation_*.json`: Saved conversations

## 🧪 Development

### Project Structure

- `src/core/`: Core business logic
- `src/ui/`: User interface components
- `src/utils/`: Utility functions and configuration
- `config/`: Configuration files and templates
- `data/`: Runtime data storage

### Adding New Features

1. **New Chatbot Specialty**: Add to `SPECIALTIES` in `chatbot.py`
2. **New Document Format**: Extend `DocumentProcessor` in `rag.py`
3. **UI Components**: Modify `streamlit_app.py`
4. **Configuration**: Update `config.yaml`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

**Ollama not found**
```bash
# Install Ollama from https://ollama.ai/
# Then pull the model:
ollama pull llama3.2:1b
```

**Import errors**
```bash
# Ensure virtual environment is activated:
source venv/bin/activate
# Reinstall requirements:
pip install -r requirements.txt
```

**Port already in use**
```bash
# Use a different port:
streamlit run src/ui/streamlit_app.py --server.port 8502
```

**Memory issues with large documents**
- Reduce chunk size in `config/config.yaml`
- Process smaller documents
- Increase system memory

### Performance Tips

- Use smaller models for faster responses
- Adjust chunk size for optimal performance
- Enable GPU acceleration if available
- Monitor memory usage with large document sets

## 📞 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the logs in `logs/`
3. Create an issue with detailed information

---

**Built with ❤️ using modern AI technologies**