let trainingData = {
  bitmaps: [],
  labels: []
};

let testingData = {
  bitmaps: [],
  labels: []
};

let trainData = []; // Training Data
let testData = []; // Testing Data

function preload() {
  trainingData.bitmaps = loadBytes('data/digits5000.bin');
  trainingData.labels = loadBytes('data/labels5000.bin');
  testingData.bitmaps = loadBytes('data/digits1000.bin');
  testingData.labels = loadBytes('data/labels1000.bin');
}

function organiseData() {
  // Training Data
  for (let n = 0; n < 5000; n++) {
    let pixelData = [];
    let offset = n * 784;
    for (let i = 0; i < 784; i++) {
      pixelData[i] = trainingData.bitmaps.bytes[i + offset] / 255;
    }
    let targets = new Array(10).fill(0);
    let answer = trainingData.labels.bytes[n];
    targets[answer] = 1;
    data = {
      inputs: pixelData,
      targets: targets
    };
    trainData[n] = data;
  }

  // Testing Data
  for (let n = 0; n < 1000; n++) {
    let pixelData = [];
    let offset = n * 784;
    for (let i = 0; i < 784; i++) {
      pixelData[i] = testingData.bitmaps.bytes[i + offset] / 255;
    }
    let targets = new Array(10).fill(0);
    let answer = testingData.labels.bytes[n];
    targets[answer] = 1;
    data = {
      inputs: pixelData,
      targets: targets
    };
    testData[n] = data;
  }
  shuffle(trainData, true);
  shuffle(testData, true);
}
