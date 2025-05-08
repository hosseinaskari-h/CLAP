let textsInput, queryInput, kInput, fetchButton;
let outputText = "";
let apiUrl = "http://127.0.0.1:5000/nearest_neighbors"; // Flask endpoint for nearest neighbors

function setup() {
  createCanvas(800, 600);
  background(240);

  // Input for dataset texts
  createP("Dataset Texts (Comma-separated):");
  textsInput = createInput("Hello world, How are you?, AI is fascinating, The future is bright, Hello AI");
  textsInput.position(20, 40);

  // Input for query text
  createP("Query Text:");
  queryInput = createInput("Hello AI");
  queryInput.position(20, 80);

  // Input for number of neighbors (k)
  createP("Number of Neighbors (k):");
  kInput = createInput("3");
  kInput.position(20, 120);

  // Fetch button
  fetchButton = createButton("Find Nearest Neighbors");
  fetchButton.position(20, 160);
  fetchButton.mousePressed(fetchNearestNeighbors);
}

function draw() {
  background(240);
  textSize(16);
  fill(0);

  // Display output
  text("Nearest Neighbors:", 20, 200);
  text(outputText, 20, 230, width - 40); // Wrap text
}

function fetchNearestNeighbors() {
  const texts = textsInput.value().split(",");
  const query = queryInput.value();
  let k = parseInt(kInput.value());

  // Adjust k dynamically
  if (k > texts.length) {
    alert(`k exceeds dataset size. Adjusting k to ${texts.length}.`);
    k = texts.length;
  }

  fetch(apiUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      texts: texts,
      query: query,
      k: k,
    }),
  })
    .then((response) => response.json())
    .then((data) => {
      let result = "Results:\n";
      for (let i = 0; i < data.indices[0].length; i++) {
        result += `${i + 1}. Neighbor: '${texts[data.indices[0][i]]}', Distance: ${data.distances[0][i].toFixed(4)}\n`;
      }
      outputText = result;
    })
    .catch((error) => {
      console.error("Error fetching nearest neighbors:", error);
      outputText = "Error fetching results. Check the console for details.";
    });
}
