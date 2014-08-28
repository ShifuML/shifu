package ml.shifu.plugin.pig.normalization;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;

import ml.shifu.core.di.spi.RequestProcessor;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.LocalDataUtils;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.pig.ExecType;
import org.apache.pig.PigServer;
import org.dmg.pmml.PMML;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class PigTransformationRequestProcessor implements RequestProcessor {

    private static Logger log = LoggerFactory
            .getLogger(PigTransformationRequestProcessor.class);
    private Configuration conf;
    private FileSystem fs;

    public void exec(Request req) throws Exception {

        Params params = req.getProcessor().getParams();
        String pathPMML = params.get("pathPMML", "model.xml").toString();
        String pathOutputActiveHeader = params.get("pathOutputActiveHeader")
                .toString();
        String modelName = (String) params.get("modelName");

        PMML pmml = PMMLUtils.loadPMML(pathPMML);

        String pluginJarFile = "target/shifu-plugin-pig-1.0-SNAPSHOT.jar";
        File pigPlugin = new File(pluginJarFile);
        if (!pigPlugin.exists()) {
            pluginJarFile = System.getenv("SHIFU_HOME")
                    + "/plugin/shifu-plugin-pig-1.0-SNAPSHOT.jar";
            if (System.getenv("SHIFU_HOME") == null)
                throw new Exception(
                        "SHIFU_HOME not set or pig jar file is missing.");
        }
        Map<String, String> pigParams = new HashMap<String, String>();

        pigParams.put("path_jar", pluginJarFile);

        String[] keys = { "pathPMML", "modelName", "delimiter", "pathInputData", "pathOutputData" };
        for (String key : keys) {
            pigParams.put(key, params.get(key).toString());
        }

        String pigScriptLocation = "src/main/pig/normalize.pig";
        File pigScript = new File(pigScriptLocation);
        if (!pigScript.exists()) {
            pigScriptLocation = System.getenv().get("SHIFU_HOME")
                    + "/plugin/modelexec.pig";
            pigScript = new File(pigScriptLocation);
            if (!pigScript.exists())
                throw new Exception("Could not load modelexec.pig");
        }
        log.info("Loading pig script from: " + pigScriptLocation);
        PigServer pigServer;
        if (Boolean.valueOf((String)params.get("localMode"))) {
            pigServer = new PigServer(ExecType.LOCAL);
        } else {
            pigServer = new PigServer(ExecType.MAPREDUCE);
        }

        pigServer.registerScript(pigScriptLocation, pigParams);

    }

}
