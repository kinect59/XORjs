

var tf = require('@tensorflow/tfjs');

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

/*
// Tiny TFJS train / predict example.
async function laod_and_test() {
 
  model = await tf.loadModel('XOR/web_model/model.json').then(model => {
  model.predict();
  
  model.summary();
  document.getElementById('micro-out-div').innerText = model.predict(tf.zeros([1,2])).dataSync();
  //y = model.predict(tf.zeros([1,2])) 
  //document.getElementById('out').innerHTML = y.dataSync()[0]
  
}

laod_and_test();
*/
