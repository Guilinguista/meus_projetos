# AI-Powered Prospect Research Pipeline

This repository contains an AI-powered pipeline that automates company research for a sales team. The pipeline takes a company website URL and a list of email addresses as input and outputs a concise summary, a slide deck, and notifications via Slack and email.

## Features
- Web scraping for company information
- AI-powered text summarization
- Slide deck generation
- Slack and email notifications
- Fully automated execution via GitHub Actions

## Usage

### Prerequisites
1. Python 3.10 or higher.
2. Install required dependencies: `pip install -r requirements.txt`.
3. Set up API keys:
   - OpenAI API Key
   - Slack API Token
   - Email credentials

### Run the Pipeline
Use the following command to execute the pipeline:
```bash
python pipeline.py --website "https://example.com" --emails "emails.txt"
