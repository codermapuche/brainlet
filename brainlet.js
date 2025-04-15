"use strict";

// ----------------------------------------------------------------------------

function biases(size) {
  return Array(size).fill(0).map(() => (Math.random() * 2 - 1));
}

function weights(length, size) {
  return Array(length).fill(size).map(biases);
}

function scheduler(step, period, minLR, maxLR) {
  const cosInner = 2 * Math.PI * (step % period) / period;
  return minLR + 0.5 * (maxLR - minLR) * (1 + Math.cos(cosInner));
}

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function dsigmoid(y) {
  return y * (1 - y);
}

function forward(input, brain) {
  var a = input;
  var activations = [a];

  for (var l = 1; l < brain.layers.length; l++) {
    var layer = brain.layers[l];
    var prev = activations[activations.length - 1];
    var out = [];

    for (var j = 0; j < layer.size; j++) {
      var sum = layer.biases[j];
      for (var i = 0; i < prev.length; i++) {
        sum += prev[i] * layer.weights[j][i];
      }
      out.push(sigmoid(sum));
    }

    activations.push(out);
  }

  return activations;
}

function train(input, target, brain, lr) {
  var activations = forward(input, brain);
  var deltas = [];

  var output = activations[activations.length - 1];
  var delta = [];
  for (var i = 0; i < output.length; i++) {
    var error = target[i] - output[i];
    delta.push(error * dsigmoid(output[i]));
  }
  deltas.unshift(delta);

  for (var l = brain.layers.length - 2; l > 0; l--) {
    var current = brain.layers[l];
    var next = brain.layers[l + 1];
    var nextDelta = deltas[0];
    var delta = [];

    for (var i = 0; i < current.size; i++) {
      var error = 0;
      for (var j = 0; j < next.size; j++) {
        error += next.weights[j][i] * nextDelta[j];
      }
      delta.push(error * dsigmoid(activations[l][i]));
    }

    deltas.unshift(delta);
  }

  for (var l = 1; l < brain.layers.length; l++) {
    var layer = brain.layers[l];
    var prev = activations[l - 1];
    var delta = deltas[l - 1];

    for (var j = 0; j < layer.size; j++) {
      for (var i = 0; i < prev.length; i++) {
        layer.weights[j][i] += lr * delta[j] * prev[i];
      }
      layer.biases[j] += lr * delta[j];
    }
  }
}

async function trainer(generator, brain, opts) {
  opts = Object.assign({
    totalSteps: Infinity,
    minLR			:     0.01,
    maxLR			:     0.10,
    lrPeriod	:   100000,
    minError	:   0.0001
  }, opts);

  let step = 0,
			iter = generator();

  while ( step < opts.totalSteps ) {
    const result = await iter.next();

    if ( result.done ) {
			break;
		}

    const data = result.value,
					lr = scheduler(step, opts.lrPeriod, opts.minLR, opts.maxLR),
					output = forward(data.input, brain).at(-1);

		let error = data.target.reduce((sum, t, i) => sum + Math.abs(t - output[i]), 0);
		error /= data.target.length;

    train(data.input, data.target, brain, lr);

    if (step % 100 === 0) {
      console.log(`Step ${step}, MSE: ${error.toFixed(6)}, LR: ${lr.toFixed(5)}`);
    }

    // Early stopping
    if ( error < opts.minError ) {
      console.log(`Converged at step ${step} with MSE ${error.toFixed(6)}`);
      break;
    }

    step++;
  }
}

function onehot(tag, tags) {
  return tags.map(id => (id === tag ? 1 : 0));
}

function ngrams(txt, n) {
  const mapa = new Map();
  for (let i = 0; i <= txt.length - n; i++) {
    const ngram = txt.slice(i, i + n);
    mapa.set(ngram, (mapa.get(ngram) || 0) + 1);
  }
  return mapa;
}

function vocab(dataset, n) {
  const vocabSet = new Set();
  for (const example of dataset) {
    const mapa = ngrams(example.input, n);
    for (const token of mapa.keys()) {
      vocabSet.add(token);
    }
  }
  return Array.from(vocabSet).sort();
}

function tensor(input, vocab) {
	const n = vocab[0].length,
				mapa = ngrams(input, n),
				total  = Array.from(mapa.values()).reduce((a, b) => a + b, 0);

  return vocab.map(token => (mapa.get(token) || 0) / (total || 1));
}

// ----------------------------------------------------------------------------

module.exports = {
	forward,
	trainer,
	weights,
	biases,
	onehot,
	ngrams,
	vocab,
	tensor
};

// ----------------------------------------------------------------------------