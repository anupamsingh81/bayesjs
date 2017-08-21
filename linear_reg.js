
include("https://rawgit.com/rasmusab/bayes.js/master/mcmc.js")


include("https://rawgit.com/rasmusab/bayes.js/master/distributions.js")

// Feel free to change this model and/or data and see what happens! :) What distributions are available can be found here: https://github.com/rasmusab/bayes.js/blob/master/distributions.js

// Setting up the data, parameter definitions and the defining the log posterior
var data = {
  x: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
  y: [-1.97, -1.07, -0.63, -0.3, 0.9, 0.84, 1.25, 1.79, 0.88, 2.17, 1.21]};
 

var params = {
  intercept: {type: "real"},
  slope: {type: "real"},
  sigma: {type: "real", lower: 0}};

var log_post = function(s, d) {
  var log_post = 0;
  // Priors
  log_post += ld.norm(s.intercept, 0, 10);
  log_post += ld.norm(s.slope, 0, 10);
  log_post += ld.cauchy(s.sigma, 0, 100); // Implicit half-cauchy
  
  // Likelihood
  for(var i = 0; i < d.y.length; i++) {
    var mu = s.intercept + s.slope * d.x[i];
    log_post += ld.norm(d.y[i], mu, s.sigma);
  }
  return log_post;
};

// Initializing the sampler and generate a sample of size 1000
var sampler =  new mcmc.AmwgSampler(params, log_post, data);
sampler.burn(2000);
var samples = sampler.sample(1);