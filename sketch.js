// CONVOLUTION LAYER TO BE ADDED

let brain;
let brainGuess;
let points = [];

function setup() {
  createCanvas(280, 280).style('border', '3px solid green');
  background(0);
  organiseData();
  brain = new NeuralNetwork(784, 64, 10);
  brainGuess = createP('').style('font-size', '32px');

  let guessButton = createButton('Guess');
  guessButton.mousePressed(() => guessDigit());
  let trainButton = createButton('Train');
  trainButton.mousePressed(() => train());
  let testButton = createButton('Test');
  testButton.mousePressed(() => testing());
  let button = createButton('Clear');
  button.mousePressed(() => {
    background(0);
    points = [];
  });
}

function draw() {
  strokeWeight(16);
  stroke(255);
  if (mouseIsPressed && mouseY < height && mouseY > 0 && mouseX < width && mouseX > 0) {
    points.push(createVector(mouseX, mouseY));
  }
  beginShape();
  noFill();
  for (let v of points) {
    vertex(v.x, v.y);
  }
  endShape();
  filter(BLUR, 4);
}

function guessDigit() {
  let img = get();
  img.resize(28, 28);
  img.loadPixels();
  let xs = [];
  for (let i = 0; i < 784; i++) {
    xs[i] = img.pixels[i * 4] / 255;
  }
  let ys = brain.predict(xs);
  let guess = oneHot(ys);
  brainGuess.html(`I think you drew the digit ${guess}.`);
}

function oneHot(ys) {
  let digit = max(ys);
  let guess = ys.indexOf(digit);
  return guess;
}

function train() {
  shuffle(trainData, true);
  for (let i = 0; i < trainData.length; i++) {
    let data = trainData[i];
    brain.train(data.inputs, data.targets);
  }
  console.log('Training Finished');
}

function testing() {
  shuffle(testData, true);
  let score = 0;
  for (let i = 0; i < testData.length; i++) {
    let data = testData[i];
    let ys = brain.predict(data.inputs);
    let guess = oneHot(ys);
    let answer = data.targets.indexOf(1);
    if (guess == answer) {
      score++;
    }
  }
  let percent = (score / testData.length) * 100;
  console.log(`${nfc(percent, 2)}%`);
}
