## **How to Run**

1. **Clone this repo and install requirements:** pip install -r requirements.txt
   
2. **Set up your OpenRouter API key** in a `.env` file or environment variable.
3. **Launch and run notebooks**
- `eda_classic_ml.ipynb:` Classical ML/EDA workflow (no LLMs agent) - Using XGBoost
- `separate_agents.ipynb:` Run Data Analyst, Feature Engineer and Explainer agents independently
- `two_agent_chain.ipynb:` Chain Data Analyst → Feature Engineering agent (LangChain) +  Training XGBoost + predictions + Explainer agent to explain predictions
- `three_agent_chain.ipynb:` Full orchestration: Analyst → Feature Engineer → Decision Support +  Training XGBoost + predictions + Explainer Agent to explain predictions

4. **Model and agent outputs** are auto-saved in the `outputs/` directory, indexed by id workflow run. e.g filename outputs/xyz_agent/00x_agentname.txt

STRUCTURE:
```
automobile/
    imports-85.data
    imports-85.names
    index
    misc
helpers/
    data_extraction.py
    eda_functions.py
    general_helpers.py
    llm_helpers.py
    models.py
outputs/
    analyst_agent/
    feature_agent/
    explanation_agent/
    decision_agent/
    2_agents_chain/
    3_agents_chain/
.env
.gitignore
requirements.txt
eda_classic_ml.ipynb
separate_agents.ipynb
two_agent_chain.ipynb
three_agent_chain.ipynb
```
