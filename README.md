# DataMining_Assignment2
Moulik Kumar | Data Mining | Assignment - 2

# Auto EDA + LLM Insights

This project performs automated exploratory data analysis (EDA) on any CSV dataset
and optionally uses an LLM to generate additional verified insights.

## Repository Structure
- src/eda.py : single-file EDA system
- logs/genai_log.md : GenAI usage log
- video/video_link.txt : demo video link

## Installation
pip install -r requirements.txt

## Run
python src/eda.py --csv path/to/data.csv --out outputs/run1 --llm openai
python src/eda.py --csv path/to/data.csv --out outputs/run_base --llm none

## Notes
- Set OPENAI_API_KEY as an environment variable to use --llm openai
- LLM only receives aggregated facts, never raw data
