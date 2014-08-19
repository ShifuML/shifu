package ml.shifu.plugin.pig.stats;

import ml.shifu.core.di.spi.RequestProcessor;
import ml.shifu.core.request.Request;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;

import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FileStatus;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.fs.PathFilter;
import org.apache.pig.ExecType;
import org.apache.pig.PigServer;
import org.codehaus.jackson.map.ObjectMapper;
import org.dmg.pmml.Model;
import org.dmg.pmml.ModelStats;
import org.dmg.pmml.PMML;
import org.dmg.pmml.UnivariateStats;
import org.jpmml.model.ImportFilter;
import org.jpmml.model.JAXBUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.xml.sax.InputSource;

import java.io.BufferedInputStream;
import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Scanner;
import java.util.zip.GZIPInputStream;

import javax.xml.transform.sax.SAXSource;
import javax.xml.transform.stream.StreamResult;

public class PigStatsRequestProcessor implements RequestProcessor {

    private static Logger log = LoggerFactory
            .getLogger(PigStatsRequestProcessor.class);
    private Configuration conf;
    private FileSystem fs;

    private Boolean localMode;

    public void exec(Request req) throws Exception {

        Params params = req.getProcessor().getParams();

        localMode = (Boolean) params.get("localMode", false);

        PigServer pigServer;

        Map<String, String> pigParams = new HashMap<String, String>();

        String[] keys = { "pathData", "delimiter", "pathPreTrainingStats" };

        for (String key : keys) {
            pigParams.put(key, params.get(key).toString());
            log.info(key + " : " + params.get(key).toString());
        }

        ObjectMapper mapper = new ObjectMapper();
        String json = mapper.writeValueAsString(req);

        pigParams.put("request", json);

        PMML pmml;

        if (localMode) {
            pigServer = new PigServer(ExecType.LOCAL);
            pmmlUpdateLocal(req);
            pmml = PMMLUtils.loadPMML((String) req.getProcessor().getParams()
                    .get("pathPMML"));
        } else {
            pigServer = new PigServer(ExecType.MAPREDUCE);
            pmmlUpdateHDFS(req);
            pmml = loadPMML((String) req.getProcessor().getParams()
                    .get("pathPMML"));
        }

        // log.info("Directory: " + System.getProperty("user.dir"));

        //look into pigUnitTest or removing @test from unit test
        String pluginJarFile = "target/shifu-plugin-pig-1.0-SNAPSHOT.jar";
        File pigPlugin = new File(pluginJarFile);
        if (!pigPlugin.exists()) {
            pluginJarFile = System.getenv("SHIFU_HOME") + "/plugin/shifu-plugin-pig-1.0-SNAPSHOT.jar";
            if (System.getenv("SHIFU_HOME") == null)
                throw new Exception(
                        "SHIFU_HOME not set or pig jar file is missing.");
        }

        pigParams.put("pig_jars", pluginJarFile);

        String pigScriptLocation = "src/main/pig/preTrainingStats.pig";
        File pigScript = new File(pigScriptLocation);
        if (!pigScript.exists()) {
            pigScriptLocation = System.getenv().get("SHIFU_HOME")
                    + "/plugin/preTrainingStats.pig";
            pigScript = new File(pigScriptLocation);
            if (!pigScript.exists())
                throw new Exception("Could not load preTrainingStats.pig");
        }
        log.info("Loading pig script from: " + pigScriptLocation);

        for (String key : pigParams.keySet()) {
            log.info(key + " : " + pigParams.get(key));
        }

        pigServer.registerScript(pigScriptLocation, pigParams);

        ModelStats modelStats = new ModelStats();

        List<Scanner> scanners = getDataScanners(
                (String) params.get("pathPreTrainingStats"), localMode);
        for (Scanner s : scanners) {
            String[] raw;
            while (s.hasNextLine()) {
                raw = s.nextLine().trim().split("\\|");
                if (req.getBindings().get(0).getSpi()
                        .equalsIgnoreCase("UnivariateStatsCalculator")) {
                    UnivariateStats univariateStats = loadStatsFromPMML(raw[1]);
                    log.info("UnivaraiteStats:  " + univariateStats);
                    modelStats.getUnivariateStats().add(univariateStats);
                }
            }
        }

        for (Model model : pmml.getModels()) {
            if (model.getModelName().equalsIgnoreCase(
                    (String) req.getBindings().get(0).getParams()
                            .get("modelName"))) {
                model.setModelStats(modelStats);
            }
        }

        savePMML(pmml, (String) params.get("pathPMML"));

    }

    public static void savePMML(PMML pmml, String path) {
        OutputStream os = null;

        try {
            os = new FileOutputStream(path);
            StreamResult result = new StreamResult(os);
            JAXBUtil.marshalPMML(pmml, result);
        } catch (Exception e) {
            log.error(e.toString());
        } finally {
            IOUtils.closeQuietly(os);
        }
    }

    public static PMML loadPMML(String path) throws Exception {

        Configuration conf = new Configuration();
        FileSystem fs = FileSystem.get(conf);
        Path pmmlFilePath = new Path(path);
        FSDataInputStream in = fs.open(pmmlFilePath);

        try {
            InputSource source = new InputSource(in);
            SAXSource transformedSource = ImportFilter.apply(source);
            return JAXBUtil.unmarshalPMML(transformedSource);

        } catch (Exception e) {
            e.printStackTrace();
            throw e;
        }
    }

    public static UnivariateStats loadStatsFromPMML(String data)
            throws Exception {

        log.info("The PMML: " + data);
        ByteArrayInputStream in = new ByteArrayInputStream(data.getBytes());
        try {
            InputSource source = new InputSource(in);
            SAXSource transformedSource = ImportFilter.apply(source);
            PMML pmml = JAXBUtil.unmarshalPMML(transformedSource);
            return pmml.getModels().get(0).getModelStats().getUnivariateStats()
                    .get(0);

        } catch (Exception e) {
            e.printStackTrace();
            throw e;
        }
    }

    private void pmmlUpdateLocal(Request req) throws Exception {

        Params params = req.getProcessor().getParams();

        File folderExisting = new File(
                (String) params.get("pathPreTrainingStats"));
        if (folderExisting.exists()) {
            FileUtils.deleteDirectory(folderExisting);
            log.info("Deleting: " + folderExisting.getPath());
        } else
            log.info("pathPreTrainingStats does not already exist.");

    }

    private void pmmlUpdateHDFS(Request req) throws Exception {

        Params params = req.getProcessor().getParams();

        conf = new Configuration();
        fs = FileSystem.get(conf);

        Path pathResults = new Path((String) params.get("pathPreTrainingStats"));
        if (fs.exists(pathResults)) {
            log.info("Deleting: " + pathResults.getName());
            fs.delete(pathResults, true);
        } else
            log.info((String) params.get("pathPreTrainingStats")
                    + " does not already exist.");

    }


    /**
     * Get the data scanners for some specified path if the file is directory,
     * get all scanner of normal sub-files if the file is normal file, get its
     * scanner !!! Notice, all hidden files (file name start with ".") will be
     * skipped !!! Warning: scanner instances should be closed by caller.
     * 
     * @param path
     *            - file path to get the scanner
     * @param sourceType
     *            - local/hdfs
     * @return scanners for specified path
     * @throws IOException
     *             - if any I/O exception in processing
     */
    public List<Scanner> getDataScanners(String path, boolean local)
            throws IOException {

        conf = new Configuration();

        if (!local) {
            fs = FileSystem.get(conf);
        } else {
            fs = FileSystem.getLocal(conf).getRaw();
        }
        FileStatus[] listStatus;
        Path p = new Path(path);
        if (fs.getFileStatus(p).isDir()) {
            // for folder we need filter pig header files
            listStatus = fs.listStatus(p, new PathFilter() {
                public boolean accept(Path path) {
                    return (!path.getName().startsWith("."));
                }
            });
        } else {
            listStatus = new FileStatus[] { fs.getFileStatus(p) };
        }

        List<Scanner> scanners = new ArrayList<Scanner>();
        for (FileStatus f : listStatus) {
            String filename = f.getPath().getName();

            if (f.isDir()) {
                log.warn(
                        "Skip - {}, since it's direcory, please check your configuration.",
                        filename);
                continue;
            }

            log.debug("Creating Scanner for file: {} ", filename);
            if (filename.endsWith(".gz")) {
                scanners.add(new Scanner(new GZIPInputStream(fs.open(f
                        .getPath()))));
            } else if (filename.endsWith(".bz2")) {
                scanners.add(new Scanner(new BZip2CompressorInputStream(fs
                        .open(f.getPath()))));
            } else {
                scanners.add(new Scanner(new BufferedInputStream(fs.open(f
                        .getPath()))));
            }
        }

        return scanners;
    }
}
