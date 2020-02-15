function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y) {
  return y * (1 - y);
}

class NeuralNetwork {
  constructor(input_nodes, hidden_nodes, output_nodes) {
    this.input_nodes = input_nodes;
    this.hidden_nodes = hidden_nodes;
    this.output_nodes = output_nodes;

    this.weights_ih = new Matrix(this.hidden_nodes, this.input_nodes);
    this.weights_ho = new Matrix(this.output_nodes, this.hidden_nodes);
    this.weights_ih.randomize();
    this.weights_ho.randomize();

    this.weights_ho_t = Matrix.transpose(this.weights_ho);

    this.bias_h = new Matrix(this.hidden_nodes, 1);
    this.bias_o = new Matrix(this.output_nodes, 1);
    this.bias_h.randomize();
    this.bias_o.randomize();

    this.learning_rate = 0.1;
  }

  predict(input_array) {
    // Computing Hidden Outputs
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    hidden.map(sigmoid); // Activation function

    // Computing Output Layer's Output!
    let outputs = Matrix.multiply(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(sigmoid);

    return outputs.toArray();
  }

  train(input_array, target_array) {
    // Computing Hidden Outputs
    let inputs = Matrix.fromArray(input_array);
    let hidden = Matrix.multiply(this.weights_ih, inputs);
    hidden.add(this.bias_h);
    hidden.map(sigmoid); // Activation function

    // Computing Output Layer's Output!
    let outputs = Matrix.multiply(this.weights_ho, hidden);
    outputs.add(this.bias_o);
    outputs.map(sigmoid); // Neural Net's Guess

    // Converting target array to matrix object
    let targets = Matrix.fromArray(target_array);

    // Calculate Error
    let output_errors = Matrix.subtract(targets, outputs);

    // Calculate Hidden Errors
    let hidden_errors = Matrix.multiply(this.weights_ho_t, output_errors);

    // Calculate gradients
    let gradients = Matrix.map(outputs, dsigmoid);
    gradients.multiply(output_errors);
    gradients.multiply(this.learning_rate);

    // Calculate Deltas
    let hidden_t = Matrix.transpose(hidden);
    let weight_ho_deltas = Matrix.multiply(gradients, hidden_t);

    // Adjust Hidden -> Output weights and output layer's biases
    this.weights_ho.add(weight_ho_deltas);
    this.bias_o.add(gradients);

    // Calculate the hidden gradients
    let hidden_gradients = Matrix.map(hidden, dsigmoid);
    hidden_gradients.multiply(hidden_errors);
    hidden_gradients.multiply(this.learning_rate);

    // Calculate Deltas
    let input_t = Matrix.transpose(inputs);
    let weight_ih_deltas = Matrix.multiply(hidden_gradients, input_t);

    // Adjust Input -> Hidden weights and hidden biases
    this.weights_ih.add(weight_ih_deltas);
    this.bias_h.add(hidden_gradients);
  }
}
