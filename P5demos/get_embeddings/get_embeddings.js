let promptInput, generateButton;
let outputText = "";
let apiUrl = "http://127.0.0.1:5000/embeddings"; // Flask endpoint for embeddings

function setup() {
  createCanvas(800, 600);
  background(240);  
  createP();
  promptInput = createInput("Hello world, How are you?, AI is fascinating");
  promptInput.position(20, 40);

  generateButton = createButton("Get Embeddings");
  generateButton.position(20, 80);
  generateButton.mousePressed(fetchEmbeddings);
}

function draw() {
  background(240);
  textSize(16);
  fill(0);
  text("Results:", 20, 120);
  text(outputText, 20, 160, width - 40);
   text("Enter Texts (Comma Separated):", 20, 20);

}

function fetchEmbeddings() {
  const texts = promptInput.value().split(",");
  fetch(apiUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ texts: texts }),
  })
    .then((response) => response.json())
    .then((data) => {
      outputText = JSON.stringify(data.embeddings, null, 2);
    })
    .catch((error) => {
      console.error("Error:", error);
      outputText = "Error fetching embeddings.";
    });
}
