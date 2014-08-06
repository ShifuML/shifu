package ml.shifu.plugin.pig;


import com.google.common.base.Joiner;
import ml.shifu.core.di.spi.RequestProcessor;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.LocalDataUtils;
import ml.shifu.core.util.Params;

import org.apache.commons.io.FileUtils;
import org.apache.pig.ExecType;
import org.apache.pig.PigServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.nio.file.Files;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;

public class PigModelExecRequestProcessor implements RequestProcessor {

    private static Logger log = LoggerFactory.getLogger(PigModelExecRequestProcessor.class);

    public void exec(Request req) throws Exception {

        Params params = req.getProcessor().getParams();

        Boolean localMode = (Boolean) params.get("localMode", false);

        PigServer pigServer;

        if (localMode) {
            pigServer = new PigServer(ExecType.LOCAL);
        } else {
            pigServer = new PigServer(ExecType.MAPREDUCE);
        }

        Map<String, String> pigParams = new HashMap<String, String>();

        String[] keys = {"delimiter", "pathData", "pathPMML", "pathResult"};

        for (String key : keys) {
            pigParams.put(key, params.get(key).toString());
            log.info(key + " : " + params.get(key).toString());
        }

        File folderExisting = new File((String)params.get("pathResult"));  
        if (folderExisting.exists()){  
        	//System.out.print("Overwrite pathResult ("+ folderExisting.getPath() + ")? (Y/n): ");
        	// Scanner in = new Scanner(System.in);
            //String answer = in.next();
            //if(!answer.toLowerCase().contains("n"))
            FileUtils.deleteDirectory(folderExisting);
            //in.close();
         	log.info("Deleting: "+folderExisting.getPath());
       }
        else log.info("Nothing to delete");

        String pathHeader = params.get("pathHeader").toString();
        String headerDelimiter = params.get("headerDelimiter").toString();
        List<String> header = LocalDataUtils.loadHeader(pathHeader, headerDelimiter);

        pigParams.put("headerString", Joiner.on(',').join(header));

        pigServer.registerScript("src/main/pig/modelexec.pig", pigParams);
    }
}
