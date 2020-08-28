/*
async function run() {
  // Create a simple model.
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 1, inputShape: [1]}));
  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
  // Generate some synthetic data for training. (y = 2x - 1)
  const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
  const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);
  // Train the model using the data.
  await model.fit(xs, ys, {epochs: 250});
  // Use the model to do inference on a data point the model hasn't seen.
  // Should print approximately 39.
  document.getElementById('micro-out-div').innerText =
      model.predict(tf.tensor2d([20], [1, 1])).dataSync();
}
run();
*/
/*
async function run() {
  // Create a simple model.
  const model = tf.sequential();
  model.add(tf.layers.dense({units: 10, activation: 'sigmoid',inputShape: [2]}));
  model.add(tf.layers.dense({units: 1, activation: 'sigmoid',inputShape: [10]}));
  // Prepare the model for training: Specify the loss and the optimizer.
  model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
  //model.compile({loss: 'meanSquaredError', optimizer: 'rmsprop'});
  // Generate some synthetic data for training. (y = 2x - 1)
  const xs = tf.tensor2d([[0,0],[0,1],[1,0],[1,1]]);
  const target_data = tf.tensor2d([[0],[1],[1],[0]]);
  // Train the model using the data.
  await model.fit(xs, ys, {epochs: 250});
  // Use the model to do inference on a data point the model hasn't seen.
  // Should print approximately 39.
  document.getElementById('micro-out-div').innerText =
      model.predict(tf.tensor2d([1, 0])).dataSync();
}
run();
*/
/*
//var tf = require('@tensorflow/tfjs');

async function train_test() {

const model = tf.sequential();
model.add(tf.layers.dense({units: 10, activation: 'sigmoid',inputShape: [2]}));
model.add(tf.layers.dense({units: 1, activation: 'sigmoid',inputShape: [10]}));

model.compile({loss: 'meanSquaredError', optimizer: 'rmsprop'});

const training_data = tf.tensor2d([[0,0],[0,1],[1,0],[1,1]]);
const target_data = tf.tensor2d([[0],[1],[1],[0]]);

for (let i = 1; i < 100 ; ++i) {
 var h = await model.fit(training_data, target_data, {epochs: 30});
   console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
}

 model.predict(training_data).print();


}

train_test();
*/


// Tiny TFJS train / predict example.
async function laod_and_test() {
 
  model = await tf.loadLayersModel('XOR/web_model/model.json').then(model => {
  
  model.summary();
  document.getElementById('micro-out-div').innerText = model.predict(tf.zeros([1,2])).dataSync();
  //y = model.predict(tf.zeros([1,2])) 
  //document.getElementById('out').innerHTML = y.dataSync()[0]
  });
}

laod_and_test();
