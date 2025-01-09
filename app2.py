# from flask import Flask, request, render_template
# from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
# import torch

# # Initialize Flask app
# app = Flask(__name__)

# # Load the tokenizer and model
# tokenizer  = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
# model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")


# # Function to get the answer from the model
# def answer_question(question, context):
#     inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
#     outputs = model(**inputs)

#     start_scores = outputs.start_logits
#     end_scores = outputs.end_logits

#     # Get the start and end positions of the answer
#     answer_start = torch.argmax(start_scores)
#     answer_end = torch.argmax(end_scores)

#     # Convert tokens to text
#     answer_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end + 1])
#     answer = tokenizer.convert_tokens_to_string(answer_tokens)

#     return answer

# # Route for the home page
# @app.route("/")
# def home():
#     return render_template("index.html")

# # Route to process the question-answering task
# @app.route("/get_answer", methods=["POST"])
# def get_answer():
#     # Get the input data from the form
#     question = request.form.get("question")
#     context = request.form.get("context")

#     if not question or not context:
#         return "Please provide both a question and a context."

#     # Get the answer using the model
#     answer = answer_question(question, context)

#     return render_template("index.html", question=question, context=context, answer=answer)

# if __name__ == "__main__":
#     app.run(debug=True)


from flask import Flask, request, render_template , jsonify
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch


# Initialize Flask app
app = Flask(__name__)

# Load the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
model = DistilBertForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")

# Function to get the answer from the model
# def answer_question(question, context):
#     inputs = tokenizer(question, context, add_special_tokens=True, return_tensors="pt")
#     outputs = model(**inputs)

#     start_scores = outputs.start_logits
#     end_scores = outputs.end_logits

#     # Get the start and end positions of the answer
#     answer_start = torch.argmax(start_scores)
#     answer_end = torch.argmax(end_scores)

#     # Convert tokens to text
#     answer_tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end + 1])
#     answer = tokenizer.convert_tokens_to_string(answer_tokens)

#     return answer

# def answer_question(question, answer_text):


#     input_ids = tokenizer.encode(question, answer_text)


#     print('Query has {:,} tokens.\n'.format(len(input_ids)))

#     sep_index = input_ids.index(tokenizer.sep_token_id)
#     num_seg_a = sep_index + 1

#     num_seg_b = len(input_ids) - num_seg_a
#     segment_ids = [0]*num_seg_a + [1]*num_seg_b
#     assert len(segment_ids) == len(input_ids)
#     # outputs = model(torch.tensor([input_ids]),
#     #                 token_type_ids=torch.tensor([segment_ids]),
#     #                 return_dict=True)
#     outputs = model(input_ids=torch.tensor([input_ids]),
#                 attention_mask=torch.tensor([attention_mask]))


#     start_scores = outputs.start_logits
#     end_scores = outputs.end_logits
#     answer_start = torch.argmax(start_scores)
#     answer_end = torch.argmax(end_scores)

#     tokens = tokenizer.convert_ids_to_tokens(input_ids)


#     answer = tokens[answer_start]

#     for i in range(answer_start + 1, answer_end + 1):


#         if tokens[i][0:2] == '##':
#             answer += tokens[i][2:]
#         else:
#             answer += ' ' + tokens[i]

#     return answer


def answer_question(question, context):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)
    input_ids = inputs["input_ids"]  # Tokenized input IDs
    attention_mask = inputs["attention_mask"]  # Attention mask

    # Forward pass through the model
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits

    # Find the start and end of the answer
    start_index = torch.argmax(start_scores)
    end_index = torch.argmax(end_scores)

    # Decode the answer from input IDs
    answer = tokenizer.decode(input_ids[0][start_index:end_index + 1])
    return answer


# Route for the home page
@app.route("/")
def home():
    return render_template("index.html")

# Route to process the question-answering task
@app.route('/answer', methods=['POST'])
def answer():
    data = request.get_json()
    context = data.get('context')
    question = data.get('question')

    # Check if the inputs are provided
    if not context or not question:
        return jsonify({"error": "Please provide both context and question"}), 400

    # Mock answer logic for now (replace with your model's prediction later)
    answer = answer_question(question, context)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)







# from flask import Flask, request, render_template, jsonify, redirect, url_for
# import fitz  # PyMuPDF for PDF handling
# from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
# import torch
# import os

# app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = 'uploads'

# # Ensure the uploads folder exists
# os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# # Load pre-trained model and tokenizer
# model_name = "distilbert-base-uncased-distilled-squad"
# tokenizer = DistilBertTokenizer.from_pretrained(model_name)
# model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# # Home route
# @app.route('/')
# def home():
#     return render_template('index.html')

# # Route to handle PDF upload
# @app.route('/upload', methods=['POST'])
# def upload_pdf():
#     if 'file' not in request.files:
#         return "No file part"

#     file = request.files['file']

#     if file.filename == '':
#         return "No selected file"

#     if file:
#         filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
#         file.save(filepath)

#         # Extract text from the PDF
#         context = extract_text_from_pdf(filepath)
#         return render_template('index.html', context=context)

# # Function to extract text from a PDF using PyMuPDF
# def extract_text_from_pdf(filepath):
#     pdf_document = fitz.open(filepath)
#     text = ""
#     for page_num in range(len(pdf_document)):
#         page = pdf_document.load_page(page_num)
#         text += page.get_text()
#     return text

# # Route to answer questions based on context
# @app.route('/answer', methods=['POST'])
# def answer():
#     question = request.form.get('question')
#     context = request.form.get('context')

#     if not question or not context:
#         return "Missing question or context"

#     # Get the answer from the model
#     answer = answer_question(question, context)
#     return jsonify({"answer": answer})

# # Function to answer the question using the model
# def answer_question(question, context):
#     inputs = tokenizer(question, context, return_tensors="pt", truncation=True, padding=True)
#     input_ids = inputs["input_ids"]
#     attention_mask = inputs["attention_mask"]

#     outputs = model(input_ids=input_ids, attention_mask=attention_mask)

#     start_scores = outputs.start_logits
#     end_scores = outputs.end_logits

#     start_index = torch.argmax(start_scores)
#     end_index = torch.argmax(end_scores)

#     answer = tokenizer.decode(input_ids[0][start_index:end_index + 1])
#     return answer

# if __name__ == '__main__':
#     app.run(debug=True)

