"use strict";

// ----------------------------------------------------------------------------

const fs 		= require("fs"),
			path 	= require("path");

// ----------------------------------------------------------------------------

const brainlet = require("../../brainlet.js");

// ----------------------------------------------------------------------------

function sanitize(input) {
  return input.toLowerCase();
}

// ----------------------------------------------------------------------------

async function main() {
	const dataset = [ ],
				tags = [ ],
				n = 3;

	fs.readdirSync('dataset').forEach((file) => {
		const target	 = path.basename(file, '.txt'),
					filepath = path.join('dataset', file);

		tags.push(target);

		let inputs = fs.readFileSync(filepath, 'utf-8');
		inputs = inputs.split('\n');
		inputs = inputs.map((input) => (input.trim()));
		inputs = inputs.map(sanitize);
		inputs = inputs.filter(Boolean);
		inputs = inputs.map((input) => ({ input, target }));

		dataset.splice.apply(dataset, [ 0, 0 ].concat(inputs));
	});

	let brain;
	
	try {
		brain = require('./brain.json');
	} catch(err) {		
		const vocab = brainlet.vocab(dataset, n);
		
		brain = {
			vocab,
			layers: [
				{ size: vocab.length }, // Input layer
				{
					size   : 16,
					weights: brainlet.weights(16, vocab.length),
					biases : brainlet.biases(16)
				},
				{
					size   : tags.length,
					weights: brainlet.weights(tags.length, 16),
					biases : brainlet.biases(tags.length)
				}
			]
		};
		
		async function* langGenerator() {
			while ( true ) {
				dataset.sort(() => Math.random() - 0.5);

				for ( const example of dataset ) {
					const input  = brainlet.tensor(example.input, brain.vocab),
								target = brainlet.onehot(example.target, tags);

					yield { input, target };
				}
			}
		}

		console.log("Entrenando...");
		await brainlet.trainer(langGenerator, brain);
		console.log("✅ Entrenamiento terminado");
		
		fs.writeFileSync('./brain.json', JSON.stringify(brain), 'utf-8');
	}

	// --------------------------------------------------------------------------
	
	let tests = fs.readFileSync('tests', 'utf-8');
	tests = tests.split('\n');
	tests = tests.map((input) => (input.trim()));
	tests = tests.map(sanitize);
	tests = tests.filter(Boolean);
	
	let solutions = fs.readFileSync('solutions', 'utf-8');
	solutions = solutions.split('\n');
	solutions = solutions.map((input) => (input.trim()));
	solutions = solutions.filter(Boolean);
	
	tests = tests.map((input, idx) => ({ 
		input	: input, 
		target: solutions[idx].split(' ').pop().toLowerCase()
	}));

  let matchs = 0;
	
	for ( const test of tests ) {
		const input = brainlet.tensor(test.input, brain.vocab),
					preview = test.input.slice(0, 30).padEnd(30);
		
		let target = brainlet.forward(input, brain).at(-1);
		target = tags[target.indexOf(Math.max(...target))];
		
    if ( target === test.target ) {
			matchs++;
			console.log(`✅ "${preview}" ${target.padEnd(8)}`);
		} else {
			console.log(`❌ "${preview}" ${target.padEnd(8)} (${test.target})`);			
		}
	}

  console.log(`Precisión: ${(matchs / tests.length * 100).toFixed(2)}%`);
}

main();
