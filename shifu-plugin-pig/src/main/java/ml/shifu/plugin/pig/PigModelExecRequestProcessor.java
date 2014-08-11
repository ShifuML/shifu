package ml.shifu.plugin.pig;

import com.google.common.base.Joiner;
import com.google.common.base.Splitter;

import ml.shifu.core.di.spi.RequestProcessor;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.LocalDataUtils;
import ml.shifu.core.util.Params;

import org.apache.commons.io.FileUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.pig.ExecType;
import org.apache.pig.PigServer;
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

public class PigModelExecRequestProcessor implements RequestProcessor {

	private static Logger log = LoggerFactory
			.getLogger(PigModelExecRequestProcessor.class);
	private Configuration conf;
	private FileSystem fs;

	public void exec(Request req) throws Exception {

		Params params = req.getProcessor().getParams();

		Boolean localMode = (Boolean) params.get("localMode", false);

		Map<String, String> pigParams = null;
		PigServer pigServer;
		
		if (localMode) {
			pigServer = new PigServer(ExecType.LOCAL);
			pigParams = execLocal(req);
		}
		else {
			pigServer = new PigServer(ExecType.MAPREDUCE);
			pigParams = execHDFS(req);
		}
		
		log.info("Directory: " + System.getProperty("user.dir"));
		
		if(System.getenv("SHIFU_HOME") != null) {
			pigParams.put("pig_jars", System.getenv("SHIFU_HOME")+"/plugin/*/*.jar");
		}
		else {
			pigParams.put("pig_jars", "shifu-core/target/shifu-core-0.3.0-SNAPSHOT/plugin/*/*.jar");
		}
		
		String pigScriptLocation = "src/main/pig/modelexec.pig";
		File pigScript = new File(pigScriptLocation);
		if (!pigScript.exists()) {
			pigScriptLocation = System.getenv().get("SHIFU_HOME")
					+ "/plugin/modelexec.pig";
			pigScript = new File(pigScriptLocation);
			if (!pigScript.exists())
				throw new Exception("Could not load modelexec.pig");
		}
		log.info("Loading pig script from: " + pigScriptLocation);
		pigServer.registerScript(pigScriptLocation, pigParams);


	}

	
	private Map<String, String> execLocal(Request req) throws Exception {

		Params params = req.getProcessor().getParams();

		Map<String, String> pigParams = new HashMap<String, String>();

		String[] keys = { "delimiter", "pathData", "pathPMML", "pathResult" };

		for (String key : keys) {
			pigParams.put(key, params.get(key).toString());
			log.info(key + " : " + params.get(key).toString());
		}

		File folderExisting = new File((String) params.get("pathResult"));
		if (folderExisting.exists()) {
			FileUtils.deleteDirectory(folderExisting);
			log.info("Deleting: " + folderExisting.getPath());
		} else
			log.info("pathResults does not already exist.");

		String pathHeader = params.get("pathHeader").toString();
		String headerDelimiter = params.get("headerDelimiter").toString();
		List<String> header = LocalDataUtils.loadHeader(pathHeader,
				headerDelimiter);

		pigParams.put("headerString", Joiner.on(',').join(header));

		return pigParams;
	}

	
	private Map<String, String> execHDFS(Request req) throws Exception {

		Params params = req.getProcessor().getParams();

		conf = new Configuration();
		fs = FileSystem.get(conf);

		Map<String, String> pigParams = new HashMap<String, String>();

		String[] keys = { "delimiter", "pathData", "pathPMML", "pathResult" };

		for (String key : keys) {
			pigParams.put(key, params.get(key).toString());
			log.info(key + " : " + params.get(key).toString());
		}

		Path pathResults = new Path((String) params.get("pathResult"));
		if (!fs.exists(pathResults)) {
			log.info("Deleting: " + pathResults.getName());
			fs.delete(pathResults, true);
		} else
			log.info("pathResults does not already exist.");

		
		String headerDelimiter = params.get("headerDelimiter").toString();
		Path headerFile = new Path((String) params.get("pathHeader"));
		
		List<String> header = readHDFSHeader(headerFile,
				headerDelimiter);

		pigParams.put("headerString", Joiner.on(',').join(header));
		
		return pigParams;
	}
	
	private List<String> readHDFSHeader( Path headerFile, String delimiter) throws Exception {
		
		List<String> headerList = new ArrayList<String>();
		
		FSDataInputStream in = fs.open(headerFile);
		Scanner scanner = new Scanner(new BufferedReader(new InputStreamReader(in)));
        while (scanner.hasNextLine()) {
            String line = scanner.nextLine();
            for (String s : Splitter.on(delimiter).split(line)) {
                headerList.add(s);
            }
        }
        in.close();
        scanner.close();
		return headerList;
	}
}
