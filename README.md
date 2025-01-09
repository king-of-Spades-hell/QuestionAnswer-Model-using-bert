# BERT-Based Question Answering Application

This is a Flask web application that uses the **DistilBERT** model for Question Answering (QA) tasks. The app takes a context and a question as input, processes the data using the `DistilBertForQuestionAnswering` model, and provides the answer.

## Features

- Accepts input in the form of a question and context via a web interface or JSON API.
- Extracts answers to questions from the given context using the **DistilBERT** model.
- Supports both text inputs and PDF uploads (for extracting context from documents).
- Handles tokenization, attention masking, and inference seamlessly.

---

## Installation

### Prerequisites
- Python 3.7 or higher
- Pip package manager

### Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/qa-flask-app.git
   cd qa-flask-app
