[<img src="images/logo/shifu.png" alt="Shifu" align="left">](http://shifu.ml)<div align="right">[![Build Status](https://travis-ci.org/ShifuML/shifu.svg?branch=master)](https://travis-ci.org/ShifuML/shifu)</div>

#

## Getting Started

Please download latest shifu [here](https://github.com/ShifuML/shifu/wiki/shifu-0.10.5-hdp-yarn.tar.gz ), and build your first model with Shifu [tutorial](https://github.com/ShifuML/shifu/wiki/Tutorial---Build-Your-First-ML-Model).

## Conference

[QCON Shanghai 2015](http://2015.qconshanghai.com/presentation/2827) [Slides](http://www.slideshare.net/pengshanzhang/large-scale-machine-learning-at-pay-pal-risk)

## What is Shifu?
Shifu is an open-source, end-to-end machine learning and data mining framework built on top of Hadoop. Shifu is designed for data scientists, simplifying the life-cycle of building machine learning models. While originally built for fraud modeling, Shifu is generalized for many other modeling domains.

Shifu provides a simple command-line interface for each step of the model building process, including

* Statistic calculation & variable selection to determine the most predictive variables in your data
* [Variable normalization](https://github.com/ShifuML/shifu/wiki/Variable%20Transform%20in%20Shifu)
* [Distributed variable selection based on sensitivity analysis](https://github.com/ShifuML/shifu/wiki/Variable%20Selection%20in%20Shifu)
* [Distributed neural network model training](https://github.com/ShifuML/shifu/wiki/Distributed%20Neural%20Network%20Training%20in%20Shifu)
* [Distributed tree ensemble model training](https://github.com/ShifuML/shifu/wiki/Distributed%20Tree%20Ensemble%20Model%20Training%20in%20Shifu)
* Post training analysis & model evaluation

Shifu’s fast Hadoop-based, distributed neural network / logistic regression / gradient boosted trees training can reduce model training time from days to hours on TB data sets. Shifu integrates with Pig workflows on Hadoop, and Shifu-trained models can be integrated into production code with a simple Java API. Shifu leverages Pig, Akka, Encog and other open source projects.

Model details about shifu can be found in our [wiki pages](https://github.com/ShifuML/shifu/wiki)

## Contributors

 - Zhanghao Hu (zhanhu@paypal.com)
 - Grahame Jastrebski (gjastrebski@paypal.com)
 - Lavar Li (lulli@paypal.com)
 - Mark Liu (yliu15@paypal.com)
 - David Zhang (pengzhang@paypal.com)
 - Xin Zhong (xinzhong@paypal.com)
 - Simon Zhang (jzhang13@paypal.com)
 - Sharma Nitin (nsharma1@paypal.com)

## Google Group

Please join [Shifu group](https://groups.google.com/forum/#!forum/shifuml) if questions, bugs or anything else.

## Copyright and License

Copyright 2012-2016, PayPal Software Foundation under the [Apache License](LICENSE.txt).
