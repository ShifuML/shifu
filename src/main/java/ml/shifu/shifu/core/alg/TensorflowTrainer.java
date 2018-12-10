/*
 * Copyright [2012-2018] PayPal Software Foundation
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
package ml.shifu.shifu.core.alg;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.Charset;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.util.Environment;

/**
 * TensorflowTrainer enables tensorflow model training in shifu.
 * <p>
 * TensorflowTrainer instance holds the model config and comlumn config and parses the model training params.
 * The {@link #train()} starts a python subprocess and passes all model training params to the train python script to
 * train a tensorflow model.
 * 
 * @author minizhuwei
 */
public class TensorflowTrainer {

    private static final Logger LOGGER = LoggerFactory.getLogger(TensorflowTrainer.class);

    private String inputDataPath;

    @SuppressWarnings("unused")
    private String outputModelDir;

    private double learningRate = 1;

    private List<Integer> seletectedColumnNums = new ArrayList<Integer>();

    private int targetColumnNum = 0;

    private List<String> actFuncs = new ArrayList<String>();

    private List<Integer> hiddenLayerNodes = new ArrayList<Integer>();

    private int hiddenLayers = 1;

    @SuppressWarnings("unused")
    private List<ColumnConfig> ccList;

    @SuppressWarnings("unused")
    private ModelConfig modelConfig;

    private PathFinder pathFinder;

    @SuppressWarnings("unused")
    private String alg;

    private char delimiter;

    @SuppressWarnings("unused")
    private String lossFunc;

    @SuppressWarnings("unused")
    private String optimizer;

    private int epoch = 100;
    
    private double validateRate = 0.2;
    
    private String modelName;

    private int weightColumnNum = -1;
    /**
     * Extract and store tensorflow training params from shifu configs.
     * 
     * @param modelConfig
     *            Shifu model config.
     * @param ccList
     *            Shifu column config.
     */
    @SuppressWarnings("unchecked")
    public TensorflowTrainer(ModelConfig modelConfig, List<ColumnConfig> ccList) {
        this.modelConfig = modelConfig;
        this.ccList = ccList;
        this.pathFinder = new PathFinder(modelConfig);

        for(int i = 0; i < ccList.size(); i++) {
            ColumnConfig cc = ccList.get(i);
            if(cc.isTarget()) {
                targetColumnNum = i;
            } else if(cc.isFinalSelect()) {
                seletectedColumnNums.add(i);
            } else if (cc.isWeight()) {
                weightColumnNum = i;
            }
        }
        if(seletectedColumnNums.size() == 0) {
            for(int i = 0; i < ccList.size(); i++) {
                ColumnConfig cc = ccList.get(i);
                if(cc.isTarget()) {
                    continue;
                }
                seletectedColumnNums.add(i);
            }
        }

        ModelTrainConf modelTrainConf = modelConfig.getTrain();
        learningRate = (double) modelTrainConf.getParams().get(CommonConstants.LEARNING_RATE);
        actFuncs = (List<String>) modelTrainConf.getParams().get(CommonConstants.ACTIVATION_FUNC);
        hiddenLayerNodes = (List<Integer>) modelTrainConf.getParams().get(CommonConstants.NUM_HIDDEN_NODES);
        hiddenLayers = hiddenLayerNodes.size();
        inputDataPath = pathFinder.getNormalizedDataPath();
        alg = (String) modelConfig.getParams().get(CommonConstants.TF_ALG);
        delimiter = modelConfig.getDataSetDelimiter().charAt(0);
        lossFunc = (String) modelTrainConf.getParams().get(CommonConstants.TF_LOSS);
        optimizer = (String) modelTrainConf.getParams().get(CommonConstants.TF_OPTIMIZER);
        epoch = modelTrainConf.getNumTrainEpochs();
        validateRate = modelTrainConf.getValidSetRate();
        modelName = modelConfig.getBasic().getName();
    }

    public void train() throws IOException {
        List<String> commands = buildCommands();
        ProcessBuilder pb = new ProcessBuilder(commands);

        pb.directory(new File("./"));
        pb.redirectErrorStream(true);
        LOGGER.info("Start trainning sub process. Commands {}", commands.toString());
        Process process = pb.start();
        StreamCollector sc = new StreamCollector(process.getInputStream());
        sc.start();

        try {
            process.waitFor();
        } catch (InterruptedException e) {
            throw new RuntimeException(e);
        } finally {
            if(sc != null) {
                sc.close();
            }
        }
    }

    /**
     * Build the tensorflow training script input params.
     * 
     * @return
     *         A list contains command all params to start python training.
     */
    public List<String> buildCommands() {
        List<String> commands = new ArrayList<String>();

        String actFuncStr = actFuncs.toString();
        actFuncStr = actFuncs.toString().substring(1, actFuncStr.length() - 1);
        actFuncStr = actFuncStr.replaceAll(",", "");
        String hiddenLayerNodesStr = hiddenLayerNodes.toString();
        hiddenLayerNodesStr = hiddenLayerNodesStr.substring(1, hiddenLayerNodesStr.length() - 1);
        hiddenLayerNodesStr = hiddenLayerNodesStr.replaceAll(",", "");
        String seletectedColumnNumsStr = seletectedColumnNums.toString();
        seletectedColumnNumsStr = seletectedColumnNumsStr.substring(1, seletectedColumnNumsStr.length() - 1);
        seletectedColumnNumsStr = seletectedColumnNumsStr.replaceAll(",", "");
        String delimiterStr = String.valueOf(delimiter);
        if((delimiter ^ '|') * (delimiter ^ '&') * (delimiter ^ '>') * (delimiter ^ '<') == 0) {
            delimiterStr = "\\" + delimiter;
        }

        String newGlibcPath = Environment.getProperty("glibc.path", "/x/home/website/glibc2.17");

        String pythonHome = Environment.getProperty("python.home");
        if(StringUtils.isEmpty(pythonHome)) {
            pythonHome = Environment.getProperty("PYTHON_HOME", "/x/home/website/python2.7");
        }
        if(StringUtils.isEmpty(pythonHome)) {
            throw new IllegalArgumentException(
                    "Please check if you set python install path to shifuconfig or -Dpython.home=...");
        }

        String hadoopHome = System.getenv("HADOOP_HOME");
        if(StringUtils.isEmpty(hadoopHome)) {
            throw new IllegalArgumentException("Please check if system env HADOOP_HOME is set well or not.");
        }

        commands.add(pathFinder.getScriptPath("bin/pytrain.sh"));
        commands.add(hadoopHome);
        commands.add(newGlibcPath);
        commands.add(System.getenv("LD_LIBRARY_PATH"));
        commands.add(System.getenv("JAVA_HOME"));
        commands.add(pythonHome);
        commands.add(pathFinder.getScriptPath("scripts/train.py"));
        commands.add("-learningRate");
        commands.add(String.valueOf(learningRate));
        commands.add("-epochnums");
        commands.add(String.valueOf(epoch));
        commands.add("-actfuncs");
        commands.add(actFuncStr);
        commands.add("-target");
        commands.add(String.valueOf(targetColumnNum));
        commands.add("-hiddenlayernodes");
        commands.add(hiddenLayerNodesStr);
        commands.add("-inputdaatapath");
        commands.add(inputDataPath);
        commands.add("-seletectedcolumnnums");
        commands.add(seletectedColumnNumsStr);
        commands.add("-alg");
        commands.add("dnn");
        commands.add("-delimiter");
        commands.add(delimiterStr);
        commands.add("-lossfunc");
        commands.add("loss");
        commands.add("-optimizer");
        commands.add("opti");
        commands.add("-validaterate");
        commands.add(String.valueOf(validateRate));
        commands.add("-modelname");
        commands.add(modelName);
        commands.add("-weightcolumnnum");
        commands.add(String.valueOf(weightColumnNum));
        
        return commands;
    }

    private static class StreamCollector extends Thread {
        /** Number of last lines to keep */
        private static final int LAST_LINES_COUNT = 100;
        /** Class logger */
        private static final Logger LOGGER = LoggerFactory.getLogger(StreamCollector.class);

        private static final Logger PYTHON_LOGGER = LoggerFactory.getLogger("TensorflowPython");

        /** Buffered reader of input stream */
        private final BufferedReader bufferedReader;
        /** Last lines (help to debug failures) */
        private final LinkedList<String> lastLines = new LinkedList<>();

        /**
         * Constructor.
         * 
         * @param is
         *            InputStream to dump to LOG.info
         */
        public StreamCollector(final InputStream is) {
            super(StreamCollector.class.getName());
            setDaemon(true);
            InputStreamReader streamReader = new InputStreamReader(is, Charset.defaultCharset());
            bufferedReader = new BufferedReader(streamReader);
        }

        @Override
        public void run() {
            readLines();
        }

        /**
         * Read all the lines from the bufferedReader.
         */
        private synchronized void readLines() {
            String line;
            try {
                while((line = bufferedReader.readLine()) != null) {
                    if(lastLines.size() > LAST_LINES_COUNT) {
                        lastLines.removeFirst();
                    }
                    lastLines.add(line);
                    PYTHON_LOGGER.info(line);
                }
            } catch (IOException e) {
                LOGGER.error("readLines: Ignoring IOException", e);
            }
        }

        public void close() {
            try {
                this.bufferedReader.close();
            } catch (IOException ignore) {
            }
        }

    }
}
