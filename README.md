# Question Answering System with BERT

This project implements a Flask-based web application that allows users to ask questions and receive answers from a given context using a fine-tuned BERT model. It supports interactive input and provides an efficient way to extract answers from text.

## 🚀 Features
- **Contextual Question Answering**: Users can provide a context and a question to get precise answers.
- **Fine-Tuned BERT Model**: Utilizes a fine-tuned BERT model for extracting answers.
- **Interactive Web Interface**: Simple and user-friendly web interface for input and output.
- **Customizable**: Supports easy integration of different fine-tuned models.

## 📂 Models Used
- **Question Answering Model**:
  - Fine-tuned BERT model for QA tasks.
  - Trained to extract answers from provided contexts.

## 🛠 Setup

### Prerequisites
- Python 3.x
- Flask
- PyTorch
- Transformers (Hugging Face library)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository.git
   cd your-repository
   ```
2. Install required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the fine-tuned BERT model and tokenizer in the specified directories.

### File Structure
- `app.py`: Main Flask application.
- `templates/index.html`: Web interface template.
- `models/`: Directory for storing fine-tuned models.

## ▶️ Usage
1. Start the Flask app:
   ```bash
   python app.py
   ```
2. Open a web browser and navigate to `http://127.0.0.1:5000/`.
3. Enter the context and a question in the interface.
4. Submit the form to receive the extracted answer.

## 📚 Key Functions

### Context Input
- Users provide a paragraph of text as the context.

### Question Answering
- Users input a question related to the provided context.
- The fine-tuned BERT model processes the input and returns the most relevant answer.

### Output
- The application displays the extracted answer below the input fields.

## 🔧 Customization
- **Models**: Replace the fine-tuned BERT model with your own fine-tuned model for QA tasks.
- **Routes**: Modify `/process` to include additional pre- or post-processing steps.
- **Answer Extraction Parameters**: Adjust the model's configuration for optimal performance.

## 🤝 Contributions
Contributions are welcome! Feel free to fork the repository, make improvements, and submit a pull request.

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

Let us know your thoughts or improvements by creating an issue or contacting us directly!
