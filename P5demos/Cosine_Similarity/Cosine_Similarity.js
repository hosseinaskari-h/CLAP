let firstInput, secondInput, generateButton;
let outputText = "";
let apiUrl = "http://127.0.0.1:5000/cosine_similarity"; // Flask endpoint for cosine similarity

function setup() {
  createCanvas(800, 600);
  background(240);

  createP("Enter First Text:");
  firstInput = createInput("Hello world");
  firstInput.position(20, 40);

  createP("Enter Second Text:");
  secondInput = createInput("How are you?");
  secondInput.position(20, 80);

  generateButton = createButton("Calculate Similarity");
  generateButton.position(20, 120);
  generateButton.mousePressed(fetchCosineSimilarity);
}

function draw() {
  background(240);
  textSize(16);
  fill(0);
  text("Cosine Similarity:", 20, 160);
  text(outputText, 20, 190, width - 40);
}

function fetchCosineSimilarity() {
  const texts = [firstInput.value(), secondInput.value()];
  fetch(apiUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts: texts }),
  })
    .then((response) => response.json())
    .then((data) => {
      outputText = "Similarity: " + data.cosine_similarity.toFixed(4);
    })
    .catch((error) => {
      console.error("Error:", error);
      outputText = "Error calculating similarity.";
    });
}
