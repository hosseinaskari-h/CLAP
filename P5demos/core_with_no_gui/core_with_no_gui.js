// API URL
let apiUrl = "http://127.0.0.1:5000";

// Mode and parameters (Set these before running the code)
let mode = "generate_with_probs"; // Options: "generate", "generate_with_probs", "beam_search", "generate_with_entropy", "calculate_perplexity"
let params = {
  max_length: 50,
  temperature: 1.0,
  top_k: 50,
  num_beams: 5, // For beam search
  repetition_penalty: 1.2, // For beam search
  length_penalty: 1.0, // For beam search
};

// Input and output
let promptInput;
let outputText = "";

function setup() {
  createCanvas(1920, 1080);
  background(240);

  // Prompt input
  createP();
  promptInput = createInput("The future of AI");
  promptInput.position(20, 40);

  // Generate button
  createButton("Generate").position(20, 100).mousePressed(() => fetchData());

}

function draw() {
  background(240);

  // Display prompt and results
  textSize(16);
  fill(0);
  text("Prompt", 20, 20);
  text("mode = " + mode , 20 , 80);
  text("Generated Text/Results:", 20, 160);
  text(outputText, 20, 190, width - 40); // Wrap text
}

function fetchData() {
  const prompt = promptInput.value();

  // Prepare payload
  let payload = { prompt, ...params };

  // Determine endpoint based on mode
  let endpoint = "";
  switch (mode) {
    case "generate":
      endpoint = "/generate";
      break;
    case "generate_with_probs":
      endpoint = "/generate_with_probs";
      break;
    case "beam_search":
      endpoint = "/generate_with_beam_search";
      break;
    case "generate_with_entropy":
      endpoint = "/generate_with_entropy";
      break;
    case "calculate_perplexity":
      endpoint = "/calculate_perplexity";
      break;
    default:
      console.error("Invalid mode selected!");
      return;
  }

  // Send POST request to the backend
  fetch(apiUrl + endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  })
    .then((response) => response.json())
    .then((data) => {
      // Display results based on mode
      if (mode === "generate" && data.generated_text) {
        outputText = data.generated_text;
      } else if (mode === "generate_with_probs" && data.token_probs) {
        outputText =
          "Generated Text: " +
          data.generated_text +
          "\nToken Probabilities: " +
          JSON.stringify(data.token_probs, null, 2);
      } else if (mode === "beam_search" && data.beam_results) {
        outputText = "Beam Search Results:\n" + data.beam_results.join("\n");
      } else if (mode === "generate_with_entropy" && data.generated_text) {
        outputText = "Generated Text: " + data.generated_text;
      } else if (mode === "calculate_perplexity" && data.perplexity) {
        outputText = "Perplexity: " + data.perplexity.toFixed(2);
      } else {
        outputText = "Unexpected response: " + JSON.stringify(data);
      }
    })
    .catch((error) => {
      console.error("Fetch Error:", error);
      outputText = "Error connecting to server.";
    });
}
