import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Tuple, Union
import os
import warnings
import logging

# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow warnings (set to "3" for errors only)

# Suppress PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning)  # Ignore all user warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore future warnings

# Suppress logging for transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

class TextGenerator:
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        Initialize the text generation model and tokenizer.
        :param model_path: Path or name of the pre-trained model.
        :param device: Device to run the model on ("cpu" or "cuda").
        """
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)

        # Set pad_token if not already defined
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def generate(
        self, prompt: str, max_length: int = 50, temperature: float = 1.0, top_k: int = 50
    ) -> str:
        """
        Generate text from a given prompt.
        :param prompt: Input prompt for text generation.
        :param max_length: Maximum length of the generated sequence.
        :param temperature: Sampling temperature for diversity.
        :param top_k: Top-k sampling parameter for token selection.
        :return: Generated text as a string.
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model.generate(
                inputs["input_ids"],
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                do_sample=True,
            )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except Exception as e:
            raise RuntimeError(f"Error during text generation: {e}")

    def generate_with_probs(self, prompt: str, max_length: int = 50, temperature: float = 1.0, top_k: int = 50) -> Tuple[str, List[dict]]:
        """
        Generate text and return token probabilities at each step.
        :param prompt: Input prompt for text generation.
        :param max_length: Maximum length of the generated sequence.
        :param temperature: Sampling temperature for diversity.
        :param top_k: Top-k sampling parameter for token selection.
        :return: Tuple containing the generated text and top token probabilities.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
        )
        token_probs = []
        for logits in outputs.scores:
            normalized_logits = logits - logits.max(dim=-1, keepdim=True).values  # Normalize logits
            probs = F.softmax(normalized_logits, dim=-1)  # Compute probabilities
            top_probs = torch.topk(probs, k=5, dim=-1)  # Get top 5 probabilities
            token_probs.append({
                "tokens": [self.tokenizer.decode([idx]) for idx in top_probs.indices.squeeze().tolist()],
                "probs": top_probs.values.squeeze().tolist()
            })

        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        return generated_text, token_probs

    def generate_with_beam_search(
        self, prompt: str, max_length: int = 50, num_beams: int = 5, top_k: int = 50, top_p: float = 0.9, repetition_penalty: float = 1.2, length_penalty: float = 1.0
    ) -> list:
        """
        Generate text using beam search with sampling for diversity.
        :param prompt: Input prompt for text generation.
        :param max_length: Maximum length of the generated sequence.
        :param num_beams: Number of beams for beam search.
        :param top_k: Top-k sampling parameter for token selection.
        :param top_p: Top-p sampling parameter for nucleus sampling.
        :param repetition_penalty: Penalize repeated tokens.
        :param length_penalty: Adjust preference for sequence length.
        :return: List of alternative generated sequences.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=num_beams,
            do_sample=True,               # Enable sampling
            top_k=top_k,                  # Add top-k sampling
            top_p=top_p,                  # Add nucleus sampling
            repetition_penalty=repetition_penalty,  # Penalize repetition
            length_penalty=length_penalty,          # Penalize/encourage length
            num_return_sequences=num_beams,         # Return all beams
        )
        return [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in outputs]


    def generate_with_entropy(
        self, prompt: str, max_length: int = 50, temperature: float = 1.0, top_k: int = 50
    ) -> dict:
        """
        Generate text and calculate entropy for each step.
        :param prompt: Input prompt for text generation.
        :param max_length: Maximum length of the generated sequence.
        :param temperature: Sampling temperature for diversity.
        :param top_k: Top-k sampling parameter for token selection.
        :return: Dictionary containing the generated text and entropy values for each step.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Initialize lists to store results
        token_probs = []
        entropies = []

        # Process each step's logits
        for logits in outputs.scores:
            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)
            token_probs.append(probs)

            # Calculate entropy for this step
            entropy = -torch.sum(probs * torch.log(probs + 1e-9), dim=-1).mean().item()
            entropies.append(entropy)

        # Decode the generated text
        generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)

        return {"generated_text": generated_text, "entropies": entropies}



    def calculate_perplexity(self, prompt: str) -> float:
        """
        Calculate perplexity for a given prompt.
        :param prompt: Input text for perplexity calculation.
        :return: Perplexity score.
        """
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            perplexity = torch.exp(torch.tensor(loss))
            return perplexity.item()
        except Exception as e:
            raise RuntimeError(f"Error during perplexity calculation: {e}")
