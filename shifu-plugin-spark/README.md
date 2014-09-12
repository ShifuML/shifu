#Shifu Spark Plugin
This code is a plugin for Shifu.ml. 
Currently it supports the operations for finding pre-training statistics for a given dataset, and for normalizing a dataset.
For using the code, install the following:
  - Hadoop
  - Spark
  - Shifu

Once Shifu is installed, the code for this plugin can be built using the maven command as follows:
	
	maven package

The tests have been disabled by default, as they require the assembled jar file to be present in the "target/" directory.
After building the jar with the above command, they can be run using:

	maven test -DskipTests=false
	
The plugin should by used as follows:
  1. Copy the assembled jar file into SHIFU_HOME/lib
  2. Create an appropriate request JSON file for using either Stats/ norm (instructions detailed below)
  3. Use the following command to execute the code for the plugin class mentioned in request.json: 
		
		shifu request.json
	
### Column Types:
This plugin supports three types of data in columns of a dataset:
  1. Continuous: Data for which mean/ variance can be computed, eg. Age of a person
  2. Categorical: Data which can be grouped into a limited number of sets using it's value eg. Nationality of a person
  3. Ordinal: Data which is not continuous but cannot be grouped into a small number of sets based on it's value, eg. ID numbers
	
## Using Spark Stats 
### Functionality:

This plugin currently supports Stats functionality for computing either the Univariate or Binomial statistics for columns in a dataset.
Univariate stats for a column are computed by considering that column in isolation. They include statistics like:
  - For continuous columns:
    - Basic numeric info: like mean, variance, sum, sum of squares
    - Counts of valid/ invalid fields in column
    - Sampling (done using a reservoir sample) over the column
  - For Categorical column:
    - Histogram over all values found in column
    - Counts of valid/ invalid fields
  - For Ordinal column:
    - Counts of valid/ invalid values in column

Binomial stats for a column are computed by considering tuples of values in the column and the accompanying target column value.
The column value will be denoted by CV and the target value by TV.
They include stats like:
  - For Continuous column: 
    - Sampling taken over (CV, TV) tuples- which can be used for binning the dataset and for computing KS, IV and WOE scores
    - Other statistics similar to univariate continuous.
  - For Categorical column:
    - Counts of TV for every value CV found in the column.
    - Other statistics similar to univariate categorical.
  - For Ordinal column:
    - Counts of valid/ invalid values in column

### Request.json for Stats:
An example of the Request JSON for using the stats functionality in the plugin:

```
	{
	    "name": "CalcStatsRequest",
	    "description": "step 2, calculate stats",
	    "processor": {
	        "spi": "RequestProcessor",
	        "impl": "ml.shifu.plugin.spark.stats.SparkCalcStatsRequestProcessor",
	        "params": {
	            "pathPMML": "./src/test/resources/stats/model.xml",
	            "pathPMMLOutput": "./src/test/resources/stats/generated/model.xml",
	            "pathRequest": "./src/test/resources/stats/spark_stats.json",
	            "pathInputData": "./src/test/resources/stats/data/wdbc.data",
	            "pathHDFSTmp": "ml/shifu/spark/stats/tmp",
	            "pathHDFSConf": "/usr/local/hadoop/etc/hadoop/",
	            "sparkHome": "/usr/local/spark",
	            "pathToJar": "./target/shifu-plugin-spark-0.3.0-SNAPSHOT-jar-with-dependencies.jar",
	            "sparkMode": "local"
	        }
	    },    
	    "bindings": [
	        {
	            "spi": "ml.shifu.plugin.spark.stats.interfaces.SparkStatsCalculator",
	            "impl": "ml.shifu.plugin.spark.stats.UnivariateStatsCalculator",
	            "params": {
	                "modelName": "demoModel",
	                "targetFieldName": "diagnosis",
	                "numBins": 10,
	                "negTags": [
	                    "B"
	                ],
	                "posTags": [
	                    "M"
	                ] 
	                    
	            }            
	        }
	
	    ]
	}
```
	
All paths, except pathHDFSTmp, are considered to be from the local filesystem if schema is absent. 
Any local files that need to be uploaded to HDFS will be copied to the pathHDFSTmp directory.
Description of the parameters:
- processor: 
  - impl: This should be set to "ml.shifu.plugin.spark.stats.SparkCalcStatsRequestProcessor" for computing stats, and to the equivalent class for normalizing a dataset.
  - params:
    - pathPMML: The path to the PMML XML file which is taken as input.
    - pathPMMLOutput: Path for output PMML which contains computed stats for the dataset.
    - pathRequest: Path to the request JSON. (The file which contains this JSON object): Required for passing parameters to the Spark Driver code.
    - pathInputData: Path to the input data, can be either local or HDFS. If local, file will be uploaded to pathHDFSTmp.
    - pathHDFSTmp: Path of HDFS temp directory, to which temporary input files will be copied if necessary. Considered to be on HDFS by default if no schema specified. Default to "hdfs://ml/shifu/stats/tmp"
    - pathHDFSConf: Path to the hadoop configuration files. Must be on the local filesystem. This path must contain the files "core-site.xml" and "hdfs-site.xml". Defaults to "/usr/lib/hadoop/etc/hadoop".
    - sparkHome: Local path to the spark installation.
    - pathToJar: Local path to the assembly jar file in SHIFU_HOME/lib.
    - sparkMode: Can be "local", "yarn-cluster" and "yarn-client". "yarn-client" mode does not support more than 2 executors. Defaults to "local".
    - sparkNumExecutors: Number of Spark Executors. Defaults to 2.
    - sparkExecutorMemory: Memory to be allotted to each executor. Defaults to 512m.
    - sparkDriverMemory: Memory to be allotted to Driver code. Defaults to 512m.
    - sparkExecutorCores: Number of cores each executor has access to. Defaults to 1.
  - bindings:
  - spi: Must be "ml.shifu.plugin.spark.stats.interfaces.SparkStatsCalculator"
  - impl: Must be "ml.shifu.plugin.spark.stats.UnivariateStatsCalculator" for computing univariate stats, "ml.shifu.plugin.spark.stats.BinomialStatsCalculator" for computing Binomial stats.
  - params:
    - modelName: The model to be used from the PMML XML for computing stats.
    - targetFieldName: Name of the target field.
    - numBins: Number for bins to be computed. Defaults to 11.
    - numQuantiles: Number of quantiles to be computed. Defaults to 11.
    - sampleSize: size of sample to be taken over a single column. Defaults to 100,000. Must be kept low if memory is limited.
    - maxHistogramSize: max size of distinct values to be considered for a histogram over a categorical column. Defaults to 10,000.
    - negTags: An array of values to be considered as "negative" values for the target column.
    - posTags: An array of values to be considered as "positive" values for the target column.


## Spark Normalization 
For normalization, the plugin requires an input data file and a PMML XML which contains a model with a LinearTransform. This can be generated by executing the TransformationCreator phase in shifu.
Normalization applies the Linear Transform defined in the PMML over each row in the input file and saves the result in an output file.

An example request JSON:

```
	{
	    "name": "ExecLocalTransformations",
	    "description": "step 5, execute transformations",
	    "processor": {
	        "spi": "RequestProcessor",
	        "impl": "ml.shifu.plugin.spark.norm.SparkModelTransformRequestProcessor",
	        "params": {
	            "modelName": "demoModel",
	            "pathPMML": "./src/test/resources/norm/model.xml",
	            "pathInputData": "./src/test/resources/norm/data/wdbc.train",
	            "pathOutputData": "./src/test/resources/norm/generated/normalized.txt",
	            "pathHDFSTmp":"ml/shifu/norm/tmp",
	            "pathHadoopConf":"/usr/local/hadoop/etc/hadoop",
	            "pathOutputActiveHeader": "./src/test/resources/norm/generated/normalized_header.txt",
	            "pathToJar": "./target/shifu-plugin-spark-0.3.0-SNAPSHOT-jar-with-dependencies.jar",
	            "pathRequest": "./src/test/resources/norm/5_transformexec.json",
	            "SparkHome": "/usr/local/spark",
	            "precision":"3",
	            "sparkMode": "local"
	        }
	    },
	
	    "bindings": [
	        {
	            "spi": "TransformationExecutor",
	            "impl": "ml.shifu.core.di.builtin.transform.DefaultTransformationExecutor"
	        }
	    ]
	}
```

Description of the parameters:
- impl: Must be "ml.shifu.plugin.spark.norm.SparkModelTransformRequestProcessor"
- params:
  - pathOutputActiveHeader: Path to the output header generated in normalization.
  - precision: The precision of fields to be saved in output. Keep this low to limit output file size.
  - All the other params are the same as explained in the stats request JSON.
- bindings:
  - impl: Must be "ml.shifu.core.di.builtin.transform.DefaultTransformationExecutor"
		

			

	
	
