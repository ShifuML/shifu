/**
 * Copyright [2012-2014] eBay Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.plugin.spark.norm;


import ml.shifu.core.di.spi.RequestProcessor;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.utils.CombinedUtils;
import ml.shifu.plugin.spark.utils.HDFSFileUtils;

import java.util.List;
import java.lang.ProcessBuilder;
import java.lang.ProcessBuilder.Redirect;
import java.lang.Process;

import org.dmg.pmml.DerivedField;
import org.dmg.pmml.PMML;

/**
 * This is the main entry point for the spark normalization plugin. The exec method
 * when called with the Request object will:
 * 	1. Unpack paths and other parameters from request object
 * 	2. Create a header file
 * 	3. Upload PMML XML and Request JSON to HDFS (if they are on local FS)
 * 	4. Create a process which calls the spark-submit script, with suitable arguments which include:
 * 		i. The path to the jar file of the spark-normalization plugin which contains all dependencies
 * 		ii. The name of the main class which is ml.shifu.plugin.spark.SparkNormalizer
 * 		iii. The arguments to SparkNormalizer- the HDFS Uri, and HDFS paths to PMML XML and Request JSON
 *
 *
 * Parameters required in Request object:
 * 	1. Path of PMML XML- local/ HDFS 
 * 	2. Path of Request object- local/ HDFS
 * 	3. Input file path- local/ HDFS
 * 	4. Output file path- local/ HDFS 
 * 	5. HDFS temp directory path- will be assumed to be HDFS
 * 	6. Output header file path- local	(HDFS TODO)
 * 	7. Path to the assembly Jar file of the plugin: local (HDFS?)
 * 	8. Path to hadoop configuration files: local
 * 	9. model name in PMML file
 * 	10. Spark home- for implementing the sparkHome/bin/spark-submit script
 * 	11. precision- for the Float fields in the normalized data file (default value 3)
 * 	12. delimiter- data delimiter (default value ",")
 * 	Spark options: (all optional)
 * 	13. Spark app name
 * 	14. Spark yarn mode (yarn-client or yarn-cluster)
 *	15. Spark number of executors (default 2)
 *	16. Spark driver memory (default 512m)
 *	17. Spark executor memory (default 512m)
 *	18. Spark cores per executor (default 1)
 *
 *	Note: 	1, 2, 3, 4 can be local/HDFS. Will be assumed to be local in absence of "hdfs:" scheme
 *	 		5 will be assumed to be HDFS path. "file:" scheme is invalid for this path.
 *			6, 7, 8 are currently only supported as local paths.
 */

public class SparkModelTransformRequestProcessor implements RequestProcessor {

    public void exec(Request req) throws Exception {
        // unpack all parameters from request object
        Params params = req.getProcessor().getParams();
        String pathPMML = (String) params.get("pathPMML", "model.xml");
        String pathRequest = (String) params.get("pathRequest", "request.xml");
        String pathHDFSTmp = (String) params.get("pathHDFSTmp",
                "ml/shifu/norm/tmp");
        // pathHDFSTmp contains the uploaded PMML and request files and the
        // output of the spark job
        String pathToJar = (String) params.get("pathToJar");
        String pathHadoopConf = (String) params.get("pathHadoopConf",
                "/usr/local/hadoop/etc/hadoop");
        String pathOutputActiveHeader = params.get("pathOutputActiveHeader")
                .toString();
        String pathOutputData = params.get("pathOutputData").toString();
        // the files created in pathHDFSTmp/output are concatenated in
        // pathOutputData to create final output file
        String pathInputData = params.get("pathInputData").toString();
        // get spark options
        String sparkMode = (String) params.get("sparkMode", "yarn-cluster");
        String sparkNumExecutors = (String) params
                .get("sparkNumExecutors", "2");
        String sparkExecutorMemory = (String) params.get("sparkExecutorMemory",
                "512m");
        String sparkDriverMemory = (String) params.get("sparkDriverMemory",
                "512m");
        String sparkExecutorCores = (String) params.get("sparkExecutorCores",
                "1");

        HDFSFileUtils hdfsUtils = new HDFSFileUtils(pathHadoopConf);
        pathHDFSTmp = hdfsUtils.relativeToFullHDFSPath(pathHDFSTmp);
        // accept spark's output in HDFS_temp_directory/output before
        // concatenating files to final output path.
        String pathOutputTmp = pathHDFSTmp + "/output";

        // delete output file and hdfs tmp file's output folder
        hdfsUtils.delete(pathOutputData);
        hdfsUtils.delete(pathOutputTmp);

        // upload PMML.xml, Request.json and input data to HDFS if on local FScs
        String pathHDFSPmml = hdfsUtils.uploadToHDFSIfLocal(pathPMML,
                pathHDFSTmp);
        String pathHDFSRequest = hdfsUtils.uploadToHDFSIfLocal(pathRequest,
                pathHDFSTmp);
        String pathHDFSInput = hdfsUtils.uploadToHDFSIfLocal(pathInputData,
                pathHDFSTmp);

        String hdfsUri = hdfsUtils.getHDFSUri();

        PMML pmml = PMMLUtils.loadPMML(pathPMML);
        List<DerivedField> activeFields = CombinedUtils.getActiveFields(pmml,
                params);
        List<DerivedField> targetFields = CombinedUtils.getTargetFields(pmml,
                params);
        CombinedUtils.writeTransformationHeader(pathOutputActiveHeader,
                activeFields, targetFields);

        // call spark-submit
        String Spark_submit = (String) params.get("SparkHome")
                + "/bin/spark-submit";
        ProcessBuilder procBuilder = new ProcessBuilder(Spark_submit,
                "--class", SparkNormalizer.class.getCanonicalName(), "--master",
                sparkMode, "--driver-memory", sparkDriverMemory,
                "--executor-memory", sparkExecutorMemory, "--num-executors",
                sparkNumExecutors, "--executor-cores", sparkExecutorCores,
                pathToJar, hdfsUri, pathHDFSInput, pathHDFSPmml,
                pathHDFSRequest);
        procBuilder.redirectErrorStream(true);
        procBuilder.redirectOutput(Redirect.INHERIT);
        System.out.println("Starting Spark job");
        Process proc = procBuilder.start();
        proc.waitFor();
        System.out.println("Job complete, now concatenating files");
        // now concatenate all files into a single file
        hdfsUtils.concat(pathOutputData, pathOutputTmp);
        // delete the tmp directory
        hdfsUtils.delete(pathHDFSTmp);

    }

}
