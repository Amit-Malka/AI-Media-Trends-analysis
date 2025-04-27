# AI Media Trends Insight Analyzer 📊

## Project Overview
AI Media Trends Insight Analyzer is an advanced analytics platform that visualizes and analyzes the impact of Artificial Intelligence on global media and communication trends. The system processes historical data to identify key trends and provides actionable insights through interactive visualizations and AI-powered analysis.

🌐 **[Live Demo: Try the app now!](https://ai-media-trends-analysis.streamlit.app/)**

## Key Features 🌟
- **Interactive Comparative Analysis** 📈
  - Compare AI adoption rates between countries or industries
  - Track performance metrics and growth patterns
  - Visualize trends through dynamic charts

- **Smart Insights Engine** 🤖
  - AI-generated insights with trend detection
  - Key performance indicators (KPIs)
  - Automated trend analysis and reporting

- **Data Visualization** 📊
  - Interactive line and bar charts
  - Comparative trend analysis
  - Custom filtering and time-based analysis

## Live Application 🚀
The application is deployed and accessible at:  
**[https://ai-media-trends-analysis.streamlit.app/](https://ai-media-trends-analysis.streamlit.app/)**

Features available in the live demo:
- Real-time data analysis and visualization
- Interactive country and industry comparisons
- AI-powered insights generation
- Dynamic filtering and trend analysis

## Tech Stack 🛠️
- 🐍 **Python 3.9+** – Core programming language for all modules
- ⚡ **Streamlit** – Fast, interactive dashboard UI for data exploration and insights
- 🧮 **Pandas, NumPy** – Data wrangling, cleaning, and numerical analysis
- 📈 **Plotly** – Interactive, publication-quality visualizations
- 🧠 **scikit-learn** – Machine learning pipelines, modeling, and evaluation
- 🤗 **Transformers (Hugging Face)** – Integration of state-of-the-art LLMs for text analysis
- 🔥 **PyTorch** – Deep learning backend for advanced model support

## Project Structure
```
ai-media-trends-insight/
├── data/
│   ├── raw/            # Raw, untouched data
│   └── processed/      # Cleaned & engineered datasets
├── notebooks/          # Jupyter notebooks (EDA, experiments)
├── outputs/
│   ├── plots/          # Generated visualizations
│   └── reports/        # AI-generated insights, summaries
├── src/                # Core production code (modules)
├── streamlit_app/      # Streamlit UI application code
├── logs/               # Log files
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
```

## Setup Instructions 🚀
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/ai-media-trends-insight.git
   cd ai-media-trends-insight
   ```
2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # On Linux/Mac:
   source .venv/bin/activate
   # On Windows:
   .venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Set up environment variables:**
   - Copy `.env.example` to `.env` and add your API keys or secrets as needed.

## How to Run Locally
1. **Navigate to the Streamlit app directory:**
   ```bash
   cd streamlit_app
   ```
2. **Launch the app:**
   ```bash
   streamlit run app.py
   ```
3. **Open your browser at:** [http://localhost:8501](http://localhost:8501)

## Usage Guide 📖
1. **Data Selection:**
   - Choose countries and industries to analyze
   - Select time periods of interest
   - Apply custom filters as needed

2. **Comparative Analysis:**
   - Select two entities (countries/industries) to compare
   - View side-by-side metrics and trends
   - Analyze growth patterns and differences

3. **Insights Review:**
   - Explore AI-generated insights
   - Review key performance metrics
   - Export data and findings

## Known Issues ⚠️
- 🌐 Internet connection is required for LLM model loading (Hugging Face)
- 💾 At least 8GB RAM recommended for smooth operation
- 🔑 Some features may require valid API keys (see `.env`)

## Future Improvements 🌟
- [ ] Advanced trend prediction modules
- [ ] Additional data source integrations
- [ ] Enhanced visualization options
- [ ] PDF report export functionality
- [ ] Real-time data updates

## Contributing 🤝
Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Raw data
https://www.kaggle.com/datasets/atharvasoundankar/impact-of-ai-on-digital-media-2020-2025

---

**For questions, suggestions, or contributions, feel free to open an issue or pull request! 🙌**
