package ml.shifu.plugin.spark.stats;
import java.lang.ProcessBuilder.Redirect;

import ml.shifu.core.di.spi.RequestProcessor;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.utils.HDFSFileUtils;

public class SparkCalcStatsRequestProcessor implements RequestProcessor{

	public void exec(Request req) throws Exception {
	    
	    Params params= req.getProcessor().getParams();
        // get paths
	    String pathPMML = params.get("pathPMML", "model.xml").toString();
        String pathRequest= params.get("pathRequest", "request.json").toString();
        String pathInputData= params.get("pathInputData").toString();
        String pathHDFSTmp= params.get("pathHDFSTmp", "hdfs://ml/shifu/stats/tmp").toString();
        String pathHadoopConf= params.get("pathHDFSConf", "/usr/lib/hadoop/etc/hadoop").toString();
        String sparkHome= params.get("sparkHome").toString();
        String pathToJar= params.get("pathToJar").toString();
        
        // get spark options
        String sparkMode = (String) params.get("sparkMode", "yarn-cluster");
        String sparkNumExecutors = (String) params.get("sparkNumExecutors", "2");
        String sparkExecutorMemory = (String) params.get("sparkExecutorMemory","512m");
        String sparkDriverMemory = (String) params.get("sparkDriverMemory", "512m");
        String sparkExecutorCores = (String) params.get("sparkExecutorCores","1");

        
        HDFSFileUtils hdfsUtils= new HDFSFileUtils(pathHadoopConf);
        
        // upload PMML, request and input to HDFS temp dir
        pathHDFSTmp= hdfsUtils.relativeToFullHDFSPath(pathHDFSTmp);
        String pathHdfsPmml= hdfsUtils.uploadToHDFSIfLocal(pathPMML, pathHDFSTmp);
        String pathHdfsRequest= hdfsUtils.uploadToHDFSIfLocal(pathRequest, pathHDFSTmp);
        String pathHdfsInput= hdfsUtils.uploadToHDFSIfLocal(pathInputData, pathHDFSTmp);
        String hdfsUri = hdfsUtils.getHDFSUri();
        // construct spark-submit command
        String pathSparkSubmit= sparkHome + "/bin/spark-submit";
        ProcessBuilder procBuilder = new ProcessBuilder(pathSparkSubmit,
                "--class", SparkStatsDriver.class.getCanonicalName(), "--master",
                sparkMode, "--driver-memory", sparkDriverMemory,
                "--executor-memory", sparkExecutorMemory, "--num-executors",
                sparkNumExecutors, "--executor-cores", sparkExecutorCores,
                pathToJar, hdfsUri, pathHdfsInput, pathHdfsPmml, pathHdfsRequest);
        
        procBuilder.redirectErrorStream(true);
        procBuilder.redirectOutput(Redirect.INHERIT);
        System.out.println("Starting Spark job");
        Process proc = procBuilder.start();
        proc.waitFor();
        
        // copy PMML to original pmml location
        String pathPMMLOutput= params.get("pathPMMLOutput", pathPMML).toString();
        hdfsUtils.copy(pathHdfsPmml, pathPMMLOutput);
        // delete the tmp directory
        hdfsUtils.delete(pathHDFSTmp);
        
	}
}
	