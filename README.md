<div align="center">

# 🤖 AgenticML

### *An Agentic Multi-Agent Machine Learning Pipeline*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-agenticml.onrender.com-blue?style=for-the-badge)](https://agenticml-latest.onrender.com)
[![Docker Hub](https://img.shields.io/badge/Docker_Hub-ghnkrk%2Fagenticml-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/ghnkrk/agenticml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

---

**Transform your data into intelligent models with minimal effort.**  
Upload a dataset → Let AI agents handle the rest → Download production-ready models.

[🎯 Live Demo](https://agenticml-latest.onrender.com) • [📦 Docker Hub](https://hub.docker.com/r/ghnkrk/agenticml) • [📖 Documentation](#-getting-started)

</div>

---

## 🌟 What is AgenticML?

AgenticML is a **full-stack, human-in-the-loop machine learning pipeline** powered by LLM agents. Instead of writing hundreds of lines of preprocessing, training, and evaluation code, you simply:

1. **Upload** your CSV dataset
2. **Review** AI-generated recommendations at each stage
3. **Download** trained models and predictions

The system uses a **hierarchical multi-agent architecture** where specialized LLM agents collaborate to analyze your data, suggest preprocessing strategies, design model architectures, and evaluate results — all while keeping you in control through intuitive human review checkpoints.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **LLM-Powered Analysis** | AI agents analyze your data and provide intelligent recommendations |
| 👁️ **Human-in-the-Loop** | Review and modify AI suggestions at every critical stage |
| 📊 **Auto-Preprocessing** | Automatic handling of missing values, encoding, scaling, and feature selection |
| 🎯 **Multi-Model Training** | Train multiple models simultaneously and compare performance |
| 📈 **Real-Time Progress** | Live WebSocket updates as your pipeline executes |
| 🔮 **One-Click Inference** | Upload test data and generate predictions instantly |
| 🐳 **Docker Ready** | Deploy anywhere with a single command |

---

## 🏗️ Architecture

AgenticML uses a **three-layer hierarchical orchestration** pattern inspired by enterprise workflow systems:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              🌐 WEB INTERFACE                                │
│                    (FastAPI + WebSocket + Static Files)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ⚡ L0 ORCHESTRATOR                                  │
│                 (Phase Controller: Prelim → Training → Summary)             │
└─────────────────────────────────────────────────────────────────────────────┘
                          /                          \
                         ▼                            ▼
┌────────────────────────────────┐    ┌────────────────────────────────────────┐
│      📊 L1 ORCHESTRATOR        │    │         🎯 L2 ORCHESTRATOR             │
│    (Data Preparation Phase)    │    │     (Model Training & Evaluation)      │
│                                │    │                                        │
│  • Dataset Registry            │    │  • Model Design (LLM-powered)          │
│  • Descriptive Analysis        │    │  • Multi-Model Training                │
│  • Statistical Analysis        │    │  • Comparative Evaluation              │
│  • Human Review: Preprocessing │    │  • Human Review: Model Selection       │
│  • Feature Engineering         │    │  • Human Review: Accept/Retrain        │
│                                │    │  • Inference Pipeline                  │
└────────────────────────────────┘    └────────────────────────────────────────┘
                                      
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          📝 SUMMARIZER NODE                                  │
│               (LLM generates comprehensive pipeline report)                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 🔄 Pipeline Flow

```mermaid
graph LR
    A[📤 Upload Dataset] --> B[🔍 Data Analysis]
    B --> C[🧹 Preprocessing Config]
    C --> D[👤 Human Review]
    D --> E[⚙️ Feature Engineering]
    E --> F[🎨 Model Design]
    F --> G[👤 Human Review]
    G --> H[🏋️ Model Training]
    H --> I[📊 Evaluation]
    I --> J[👤 Accept/Retrain]
    J --> K[🔮 Inference]
    K --> L[📥 Download Results]
```

---

## 🚀 Getting Started

### Option 1: Live Demo (Fastest)

Try it instantly without any installation:

👉 **[https://agenticml-latest.onrender.com](https://agenticml-latest.onrender.com)**

---

### Option 2: Docker Hub (Recommended)

Pull and run the pre-built container:

```bash
# Pull the latest image
docker pull ghnkrk/agenticml:updated

# Run the container
docker run -d -p 8000:8000 \
  -e GROQ_API_KEY=your_groq_api_key_here \
  --name agenticml \
  ghnkrk/agenticml:updated

# Access at http://localhost:8000
```

---

### Option 3: Clone from GitHub

For development or customization:

```bash
# Clone the repository
git clone https://github.com/Ghnkrk/AutoAgenticML.git
cd AutoAgenticML

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install uv
uv sync

# Set up environment variables
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# Run the application
python backend/api_server.py

# Access at http://localhost:8000
```

---

### Option 4: Docker Compose (Full Stack)

```bash
# Clone the repo
git clone https://github.com/Ghnkrk/AutoAgenticML.git
cd AutoAgenticML

# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# Build and run
docker-compose up --build

# Access at http://localhost:8000
```

---

## 📋 Prerequisites

| Requirement | Version | Purpose |
|------------|---------|---------|
| Python | 3.12+ | Runtime environment |
| Groq API Key | - | LLM inference ([Get one free](https://console.groq.com)) |
| Docker | 20.10+ | Containerization (optional) |

---

## 🎮 Usage Guide

### Step 1: Upload Your Dataset

- Navigate to the home page
- Drag & drop your CSV file or click to browse
- Specify the **target column** (what you want to predict)
- Select the task type: `Binary Classification`, `Multiclass`, or `Regression`

### Step 2: Review AI Recommendations

The system will analyze your data and present preprocessing recommendations:

- **Feature Selection**: Which columns to keep, drop, or transform
- **Encoding Strategy**: One-hot, ordinal, or target encoding
- **Scaling Method**: Standard, MinMax, or Robust scaling
- **Dimensionality Reduction**: PCA configuration

*Modify any settings before proceeding!*

### Step 3: Model Selection

The AI suggests optimal models based on your data characteristics:

- Review suggested models and their hyperparameters
- Click on "⚙️ Edit Hyperparameters" to customize
- Uncheck models you don't want to train

### Step 4: Training & Evaluation

Watch real-time progress as models are trained:

- Live logs show training status
- Performance metrics are displayed upon completion
- Review model rankings and recommendations

### Step 5: Download Results

After completion:

- **Download trained models** (`.pkl` files)
- **Download model metadata** (JSON with metrics)
- **Run inference** on new data
- **Download predictions** (CSV)

---

## 🛠️ Tech Stack

<table>
<tr>
<td align="center" width="150">

**Backend**

![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![WebSocket](https://img.shields.io/badge/-WebSocket-010101?style=flat&logo=websocket&logoColor=white)

</td>
<td align="center" width="150">

**ML/AI**

![LangChain](https://img.shields.io/badge/-LangChain-121212?style=flat)
![LangGraph](https://img.shields.io/badge/-LangGraph-4A154B?style=flat)
![Scikit-learn](https://img.shields.io/badge/-Sklearn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

</td>
<td align="center" width="150">

**Frontend**

![HTML5](https://img.shields.io/badge/-HTML5-E34F26?style=flat&logo=html5&logoColor=white)
![JavaScript](https://img.shields.io/badge/-JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)
![TailwindCSS](https://img.shields.io/badge/-Tailwind-06B6D4?style=flat&logo=tailwindcss&logoColor=white)

</td>
<td align="center" width="150">

**Infrastructure**

![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/-Render-46E3B7?style=flat&logo=render&logoColor=white)
![Groq](https://img.shields.io/badge/-Groq-FF6B6B?style=flat)

</td>
</tr>
</table>

---

## 📁 Project Structure

```
AgenticML/
├── 🐳 Dockerfile              # Container configuration
├── 📦 compose.yaml            # Docker Compose setup
├── 📋 pyproject.toml          # Python dependencies
│
├── 🔧 backend/
│   ├── api_server.py          # FastAPI application
│   ├── pipeline_manager.py    # Pipeline orchestration
│   └── pipeline_wrapper.py    # Node execution wrapper
│
├── 🎨 frontend/
│   ├── index.html             # Upload page
│   ├── pipeline.html          # Pipeline execution view
│   ├── css/styles.css         # Custom styling
│   └── js/
│       ├── app.js             # Upload logic
│       ├── pipeline.js        # Pipeline UI
│       ├── modals.js          # Human review modals
│       └── websocket.js       # Real-time updates
│
├── 🤖 Core ML Components
│   ├── main.py                # LangGraph state definition
│   ├── Nodes.py               # Pipeline node implementations
│   ├── HumanNodes.py          # Human interaction nodes
│   ├── Orchestrators.py       # L0/L1/L2 orchestrators
│   ├── promptTemplate.py      # LLM prompts
│   └── ModelResponseSchema.py # Pydantic schemas
│
└── 📊 Processing Modules
    ├── descriptive.py         # Statistical analysis
    ├── analysis.py            # Feature analysis
    ├── preprocess.py          # Data preprocessing
    ├── trainer.py             # Model training
    └── evaluator.py           # Model evaluation
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | Your Groq API key for LLM inference |

---

## 🤝 Contributing

Contributions are welcome! Feel free to:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

### Built with ❤️ using Python, FastAPI, LangGraph, and Groq

**[⬆ Back to Top](#-agenticml)**

</div>
