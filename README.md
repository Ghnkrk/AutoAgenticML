<div align="center">

# 🤖 AutoAgenticML

### *Agent-Orchestrated Machine Learning Pipeline*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-agenticml.onrender.com-blue?style=for-the-badge)](https://agenticml-latest.onrender.com)
[![Docker Hub](https://img.shields.io/badge/Docker_Hub-ghnkrk%2Fagenticml-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://hub.docker.com/r/ghnkrk/agenticml)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)

---

**Automates the journey from raw dataset to trained models — with explicit control points for human decision-making.**

The project focuses on **system design**, **state-driven orchestration**, and **ML workflow correctness**, rather than AutoML black-box optimization.

[🎯 Live Demo](https://agenticml-latest.onrender.com) • [📦 Docker Hub](https://hub.docker.com/r/ghnkrk/agenticml) • [📖 Documentation](#-getting-started)

</div>

---

## 🔍 What This Project Is

AutoAgenticML implements a **structured, multi-stage ML pipeline** using agent-style orchestration:

| Stage | Description |
|-------|-------------|
| 📥 **Ingestion** | Dataset ingestion and registration |
| 📊 **Profiling** | Descriptive statistics and dataset profiling |
| 🔬 **Analysis** | Statistical analysis (missingness, cardinality, correlation, multicollinearity) |
| 👤 **Human Review** | Human-in-the-loop preprocessing decisions |
| ⚙️ **Engineering** | Feature engineering and preprocessing execution |
| 🎯 **Selection** | Model selection using constrained, explainable model pools |
| 🏋️ **Training** | Model training, evaluation, and ranking |
| 🔄 **Retraining** | Optional retraining loops |
| 🔮 **Inference** | Inference support for unseen datasets |

> The pipeline is **deterministic, inspectable, and debuggable**, with every major decision surfaced explicitly.

---

## 🧠 Design Philosophy

This project is built around a few core principles:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  ❌  No hidden magic — every step is logged and explainable                 │
│  👤  Human-in-the-loop by design, not as an afterthought                    │
│  📊  State-driven orchestration, not conversational agents                  │
│  🧩  Separation of concerns between layers                                  │
│  🎯  Practical ML, not leaderboard chasing                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

**The goal is not to replace ML engineers, but to formalize the workflow they already follow.**

---

## 🏗️ Architecture Overview

The system uses **hierarchical orchestration** with three layers:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              🌐 WEB INTERFACE                                │
│                    (FastAPI + WebSocket + Static Files)                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          ⚡ L0 ORCHESTRATOR                                  │
│              Controls transitions between major pipeline phases             │
└─────────────────────────────────────────────────────────────────────────────┘
                          /                          \
                         ▼                            ▼
┌────────────────────────────────┐    ┌────────────────────────────────────────┐
│      📊 L1 ORCHESTRATOR        │    │         🎯 L2 ORCHESTRATOR             │
│                                │    │                                        │
│  Manages:                      │    │  Handles:                              │
│  • Dataset analysis            │    │  • Model selection                     │
│  • Statistical profiling       │    │  • Training execution                  │
│  • Preprocessing decisions     │    │  • Evaluation & ranking                │
│  • Feature preparation         │    │  • Retraining loops                    │
│                                │    │  • Inference pipeline                  │
└────────────────────────────────┘    └────────────────────────────────────────┘
```

> Each phase is driven by **explicit state transitions** rather than free-form reasoning.

---

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
    I --> J[👤 Accept/Retrain] --> G[👤 Human Review]
    J --> K[🔮 Inference] --> C[🧹 Preprocessing Config]
    K --> L[📥 Download Results]
```

## Graph Structure

<img width="1904" height="186" alt="graph_visual" src="https://github.com/user-attachments/assets/bf1f6e07-9993-495b-b22c-f70f9cd14690" />


## ⚙️ Core Components

### 📊 Analysis Layer

| Analysis Type | Purpose |
|--------------|---------|
| Missing Values | Identify data gaps and imputation needs |
| Cardinality | Detect high/low cardinality features |
| Feature–Target Correlation | Identify predictive features |
| Feature–Feature Correlation | Detect redundancy |
| Multicollinearity (VIF) | Prevent coefficient instability |
| Task Type Inference | Binary / Multiclass / Regression |

### 🧩 Preprocessing Layer

- Feature inclusion / exclusion
- Missing value handling strategies
- Encoding strategy selection (one-hot, ordinal, target)
- Scaling methods (standard, minmax, robust)
- Optional dimensionality reduction (PCA)
- Train/test–aware preprocessing logic

### 🤖 Modeling Layer

- **Constrained model pool** — no AutoML black boxes
- **Conservative default hyperparameters** — explainable baselines
- **Explicit model comparison** — transparent ranking
- **Metric-based evaluation** — F1, accuracy, precision, recall, ROC-AUC

### 🧑‍💻 Human-in-the-Loop

Human review points exist at critical decision boundaries:

| Checkpoint | User Action |
|------------|-------------|
| **Preprocessing Config** | Review/modify feature handling, scaling, encoding |
| **Model Selection** | Edit hyperparameters, remove models |
| **Evaluation Review** | Accept models or trigger retraining |
| **Inference Decision** | Choose to run predictions on new data |

---

## 🚀 Getting Started

### Option 1: Live Demo (Instant)

Try it without any installation:

👉 **[https://agenticml-latest.onrender.com](https://agenticml-latest.onrender.com)**

---

### Option 2: Docker Hub (Recommended)

```bash
# Pull the image
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

### Option 4: Docker Compose

```bash
# Clone and navigate
git clone https://github.com/Ghnkrk/AutoAgenticML.git
cd AutoAgenticML

# Create .env file
echo "GROQ_API_KEY=your_groq_api_key_here" > .env

# Build and run
docker-compose up --build

# Access at http://localhost:8000
```

---

## 🔑 Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GROQ_API_KEY` | ✅ | Your Groq API key for LLM inference ([Get one free](https://console.groq.com)) |

---

## 🧪 Example Use Case

The pipeline has been validated using the **Titanic dataset**, demonstrating:

- ✅ Correct statistical analysis
- ✅ Reasonable preprocessing decisions
- ✅ Sensible model selection
- ✅ Competitive performance without hyperparameter tuning
- ✅ Controlled retraining loops

> This serves as a **reference implementation**, not a benchmark claim.

---

## 📁 Project Structure

```
AutoAgenticML/
├── 🐳 Dockerfile              # Container configuration
├── 📦 compose.yaml            # Docker Compose setup
├── 📋 pyproject.toml          # Python dependencies
│
├── 🔧 backend/
│   ├── api_server.py          # FastAPI application
│   ├── pipeline_manager.py    # Pipeline state management
│   └── pipeline_wrapper.py    # Node execution wrapper
│
├── 🎨 frontend/
│   ├── index.html             # Dataset upload page
│   ├── pipeline.html          # Pipeline execution view
│   ├── css/styles.css         # Custom styling
│   └── js/
│       ├── app.js             # Upload logic
│       ├── pipeline.js        # Pipeline UI controller
│       ├── modals.js          # Human review modals
│       └── websocket.js       # Real-time updates
│
├── main.py                # LangGraph state definition
├── Orchestrators.py       # L0/L1/L2 orchestrators
├── promptTemplate.py      # LLM prompts
├── Nodes.py               # All pipeline node implementations
├── HumanNodes.py          # Human interaction nodes
├── ModelResponseSchema.py # Pydantic schemas
├── descriptive.py         # Statistical profiling
├── analysis.py            # Feature analysis
├── preprocess.py          # Data preprocessing
├── trainer.py             # Model training
└── evaluator.py           # Model evaluation
```

---

## 🛠️ Tech Stack

<table>
<tr>
<td align="center" width="150">

**Backend**

![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat&logo=fastapi&logoColor=white)
![Python](https://img.shields.io/badge/-Python-3776AB?style=flat&logo=python&logoColor=white)
![WebSocket](https://img.shields.io/badge/-WebSocket-010101?style=flat)

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

## 🚧 Scope & Limitations

This project intentionally operates within defined boundaries:

| Included | Not Included |
|----------|--------------|
| ✅ Structured, transparent pipelines | ❌ Black-box AutoML |
| ✅ Human oversight at key decisions | ❌ Autonomous optimization |
| ✅ Classical ML models | ❌ Deep learning |
| ✅ Clarity and correctness | ❌ Leaderboard performance |
| ✅ Educational/prototype use | ❌ Production-scale deployment |

---

## 👤 Authorship & AI Usage

This project was developed using an **AI-assisted engineering workflow**.

**The author is responsible for:**
- Overall system architecture
- Agent orchestration design
- ML analysis and preprocessing logic
- State schema design
- Training, evaluation, and retraining flow
- Dockerization and deployment

Frontend UI scaffolding and backend boilerplate were generated with AI assistance and then **integrated, validated, and adapted** by the author.

> AI tools were used as **productivity aids**, not as autonomous system designers.

---

## 📌 Status

<div align="center">

### ✅ Completed – Functional Prototype

</div>

**Future improvements may include:**
- 📊 Persistent experiment tracking
- 🔄 Expanded inference workflows
- 📈 Multi-dataset comparison
- 🎯 Advanced evaluation strategies

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

<div align="center">

### Built with clarity, correctness, and control in mind.

**[⬆ Back to Top](#-autoagenticml)**

</div>
