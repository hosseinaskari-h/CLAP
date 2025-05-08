let promptInput, generateButton, modeDropdown, deviceDropdown;
let tempInput, maxLengthInput, topKInput, numBeamsInput, repetitionPenaltyInput;
let outputText = "";
let apiUrl = "http://127.0.0.1:5000"; // Adjust this if Flask runs elsewhere
let mode = "generate";
let device = "cpu";

function setup() {
  createCanvas(800, 800); // Increased height for larger text area
  background(240);

  // Prompt input
  createP("Prompt:");
  promptInput = createInput("The future of AI");
  promptInput.position(20, 40);

  // Mode selection
  createP("Mode:");
  modeDropdown = createSelect();
  modeDropdown.option("Generate Text");
  modeDropdown.option("Generate with Probs");
  modeDropdown.option("Beam Search");
  modeDropdown.option("Generate with Entropy");
  modeDropdown.option("Calculate Perplexity");
  modeDropdown.changed(() => (mode = modeDropdown.value()));
  modeDropdown.position(20, 80);

  // Device selection
  createP("Device:");
  deviceDropdown = createSelect();
  deviceDropdown.option("CPU");
  deviceDropdown.option("GPU");
  deviceDropdown.changed(() => (device = deviceDropdown.value().toLowerCase()));
  deviceDropdown.position(20, 120);

  // Parameter inputs
  createP("Temperature:");
  tempInput = createInput("1.0");
  tempInput.position(20, 160);

  createP("Max Length:");
  maxLengthInput = createInput("50");
  maxLengthInput.position(20, 200);

  createP("Top K:");
  topKInput = createInput("50");
  topKInput.position(20, 240);

  createP("Number of Beams (Beam Search):");
  numBeamsInput = createInput("5");
  numBeamsInput.position(20, 280);

  createP("Repetition Penalty (Beam Search):");
  repetitionPenaltyInput = createInput("1.2");
  repetitionPenaltyInput.position(20, 320);

  // Generate button
  generateButton = createButton("Generate");
  generateButton.position(20, 360);
  generateButton.mousePressed(() => fetchData());

  // Instructions
  textSize(12);
  textAlign(LEFT);
  fill(0);
  text(
    "Adjust parameters and modes to experiment with the text generation model.",
    20,
    400
  );
}

function draw() {
  background(240);

  // Display prompt and results
  textSize(16);
  fill(0);
  text("Prompt: " + promptInput.value(), 20, 20);
  text("Generated Text:", 20, 440);

  // Wrap text for better readability
  text(outputText, 20, 470, width - 40); 
}

function fetchData() {
  const prompt = promptInput.value();
  const maxLength = parseInt(maxLengthInput.value());
  const temperature = parseFloat(tempInput.value());
  const topK = parseInt(topKInput.value());
  const numBeams = parseInt(numBeamsInput.value());
  const repetitionPenalty = parseFloat(repetitionPenaltyInput.value());

  // Determine API endpoint and payload based on mode
  let endpoint = "";
  let payload = { prompt, max_length: maxLength, temperature, top_k: topK };

  switch (mode) {
    case "Generate Text":
      endpoint = "/generate";
      break;
    case "Generate with Probs":
      endpoint = "/generate_with_probs";
      break;
    case "Beam Search":
      endpoint = "/generate_with_beam_search";
      payload.num_beams = numBeams;
      payload.repetition_penalty = repetitionPenalty;
      break;
    case "Generate with Entropy":
      endpoint = "/generate_with_entropy";
      break;
    case "Calculate Perplexity":
      endpoint = "/calculate_perplexity";
      delete payload.temperature; // Perplexity doesn't use temperature
      delete payload.top_k; // Perplexity doesn't use top_k
      break;
  }

  // Add device info
  payload.device = device;

  // Fetch data from the Flask API
  fetch(apiUrl + endpoint, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
  })
    .then((response) => response.json())
    .then((data) => {
      if (data.generated_text) {
        outputText = data.generated_text;
      } else if (data.perplexity) {
        outputText = "Perplexity: " + data.perplexity.toFixed(2);
      } else if (data.beam_results) {
        outputText = "Beam Search Results:\n" + data.beam_results.join("\n");
      } else if (data.token_probs) {
        outputText = "Generated Text: " + data.generated_text; // For "Generate with Probs"
      } else {
        outputText = "Unexpected response: " + JSON.stringify(data);
      }
    })
    .catch((error) => {
      console.error("Fetch Error:", error);
      outputText = "Error connecting to server.";
    });
}
