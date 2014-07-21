### Spark Overview
* a fast and general-purpose cluster computing system
* supports a set of higher-level tools including MLlib for machine learning
* resilient distributed dataset (RDD) is a collection of elements partitioned across cluster nodes 
** can be operated on in parallel
** provides fault tolerance
* MLlib: LogisticRegression, linear SVM, Decision Tree

### Dependency
1. Spark.core_2.10 version 1.0.0
2. Spark.mllib_2.10 version 1.0.0
3. guava version version 17.0 (note: shifu-core guava 14.0.0)
4. exclude akka 2.1.1 which is inherited from shift-core

### Get started
1. Initialize Spark

        SparkConf conf = new SparkConf().setAppName(appName).setMaster(master);
        JavaSparkContext sc = new JavaSparkContxt(conf);`

2. Prepare Dataset

 From Parallelized Collections

        List<Integer> data = Arrays.asList(1, 2, 3, 4, 5);
        JavaRDD<Integer> distData = sc.parallelize(data);

 From External DataSets

        JavaRDD<String> distFile = sc.textFile("data.txt");   
  OR

         JavaRDD<String> distFile = sc.textFile("hdfs://data.txt");

3. Passing Functions to Spark

        class ParseLabeledPoint implements Function<String, LabeledPoint> {
  	         public LabeledPoint call(String s) {...
                   	for (int i = 0; i < len; i++) {
                             x[i] = Double.parseDouble(tokens[i]);
                  }
              return new LabeledPoint(y, Vectors.dense(x));
               }
          }
         RDD<LabeledPoint> data = distData.map(new ParseLabeledPoint().cache().rdd();
4.  Train LogisticRegressionModel

            /*
            * @param input RDD of (label, array of features) pairs.
            * @param numIterations Number of iterations of gradient descent to run.
            * @param stepSize Step size to be used for each iteration of gradient descent.
            * @param miniBatchFraction Fraction of data to be used per iteration.
            */
            LogisticRegressionModel lrModel = LogisticRegressionWithSGD.train(data, iterations,stepSize,miniBatchFraction);

**Notice that the train errors are printed out via log for the the last 10 iterations only. **

5. Calculate Evaluation Score

 Prepare DataSet and calculate score

            JavaRDD<Vector> evalVectors = lines.map(new ParseVector().cache();
            List<Double> evalList = lrModel.predict(evalVectors).cache().collect();

 Calculate Evaluation Metrics

            val scoreAndLabels = test.map { point => val score = model.predict(point.features)(score, point.label)}
            // Get evaluation metrics.
            val metrics = new BinaryClassificationMetrics(scoreAndLabels)
            val auROC = metrics.areaUnderROC()

   precision, recall, F-measure, precision-recall curve 
 `pr(),  precisionByThreshold(),recallByThreshold()..`

   area under the curves (AUC) - `areaUnderPR()`

    receiver operating characteristic (ROC) - `areaUnderROC(), roc()`
