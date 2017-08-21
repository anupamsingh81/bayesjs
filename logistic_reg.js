include("https://rawgit.com/rasmusab/bayes.js/master/mcmc.js")


include("https://rawgit.com/rasmusab/bayes.js/master/distributions.js")


//A small demo of https://github.com/rasmusab/bayes.js running a standard Normal model. Just click "start sampling" to eh... start sampling...

// Feel free to change this model and/or data and see what happens! :) What distributions are available can be found here: https://github.com/rasmusab/bayes.js/blob/master/distributions.js

// Setting up the data, parameter definitions and the defining the log posterior
// The heights of the last ten American presidents in cm, from Kennedy to Obama 
var data = [183, 192, 182, 183, 177, 185, 188, 188, 182, 185];

var params = {
  mu: {type: "real"},
  sigma: {type: "real", lower: 0}};

var log_post = function(state, data) {
  var log_post = 0;
  // Priors
  log_post += ld.norm(state.mu, 0, 100);
  log_post += ld.unif(state.sigma, 0, 100);
  // Likelihood
  for(var i = 0; i < data.length; i++) {
    log_post += ld.norm(data[i], state.mu, state.sigma);
  }
  return log_post;
};

// Initializing the sampler and generate a sample of size 1000
var sampler =  new mcmc.AmwgSampler(params, log_post, data);
sampler.burn(500);
var samples = sampler.sample(1);
