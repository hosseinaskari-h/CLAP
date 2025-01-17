from flask import Flask, request, jsonify
from text_generation import TextGenerator
from flask_cors import CORS


# Initialize Flask app and text generator
app = Flask(__name__)
text_generator = TextGenerator(model_path="gpt2", device="cpu")
CORS(app)  # Enable CORS for all routes


@app.route("/generate", methods=["POST"])
def generate():
    data = request.json
    prompt = data.get("prompt", "")
    max_length = data.get("max_length", 50)
    temperature = data.get("temperature", 1.0)
    top_k = data.get("top_k", 50)
    try:
        result = text_generator.generate(prompt, max_length, temperature, top_k)
        return jsonify({"generated_text": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate_with_probs", methods=["POST"])
def generate_with_probs():
    data = request.json
    prompt = data.get("prompt", "")
    max_length = data.get("max_length", 50)
    temperature = data.get("temperature", 1.0)
    top_k = data.get("top_k", 50)
    try:
        result, token_probs = text_generator.generate_with_probs(prompt, max_length, temperature, top_k)
        return jsonify({"generated_text": result, "token_probs": token_probs})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate_with_beam_search", methods=["POST"])
def generate_with_beam_search():
    data = request.json
    prompt = data.get("prompt", "")
    max_length = data.get("max_length", 50)
    num_beams = data.get("num_beams", 5)
    repetition_penalty = data.get("repetition_penalty", 1.2)
    length_penalty = data.get("length_penalty", 1.0)
    try:
        results = text_generator.generate_with_beam_search(prompt, max_length, num_beams, repetition_penalty=repetition_penalty, length_penalty=length_penalty)
        return jsonify({"beam_results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/generate_with_entropy", methods=["POST"])
def generate_with_entropy():
    data = request.json
    prompt = data.get("prompt", "")
    max_length = data.get("max_length", 50)
    temperature = data.get("temperature", 1.0)
    top_k = data.get("top_k", 50)
    try:
        result = text_generator.generate_with_entropy(prompt, max_length, temperature, top_k)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/calculate_perplexity", methods=["POST"])
def calculate_perplexity():
    data = request.json
    prompt = data.get("prompt", "")
    try:
        perplexity = text_generator.calculate_perplexity(prompt)
        return jsonify({"perplexity": perplexity})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
