package ml.shifu.shifu.core.alg;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.ProcessBuilder.Redirect;
import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.fs.PathFinder;

public class TensorflowTrainer {

    private static final Logger LOGGER = LoggerFactory.getLogger(TensorflowTrainer.class);

    private String inputDataPath;

    private String outputModelDir;

    private double learningRate = 1;

    private List<Integer> seletectedColumnNums = new ArrayList<Integer>();

    private int targetColumnNum = 0;

    private List<String> actFuncs = new ArrayList<String>();

    private List<Integer> hiddenLayerNodes = new ArrayList<Integer>();

    private int hiddenLayers = 1;

    private List<ColumnConfig> ccList;

    private ModelConfig modelConfig;

    private PathFinder pathFinder;

    private String alg;

    private char delimiter;

    private String lossFunc;

    private String optimizer;
    
    private int epoch = 100;

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
        alg = modelConfig.getAlgorithm();
        delimiter = modelConfig.getDataSetDelimiter().charAt(0);
        lossFunc = (String) modelTrainConf.getParams().get(CommonConstants.TF_LOSS);
        optimizer = (String) modelTrainConf.getParams().get(CommonConstants.TF_OPTIMIZER);
        epoch = modelTrainConf.getNumTrainEpochs();
    }

    public void train() throws IOException {
        List<String> commands = buildCommands();
        ProcessBuilder pb = new ProcessBuilder(commands);
        pb.directory(new File("./"));
        LOGGER.info("Start trainning sub process. Commands {}", commands.toString());
        Process p = pb.start();
    }

    @SuppressWarnings("unchecked")
    public List<String> buildCommands() {
        List<String> commands = new ArrayList<String>();

        String actFuncStr = actFuncs.toString();
        actFuncStr = actFuncs.toString().substring(1, actFuncStr.length() - 1);
        String hiddenLayerNodesStr = hiddenLayerNodes.toString();
        hiddenLayerNodesStr = hiddenLayerNodesStr.substring(1, hiddenLayerNodesStr.length() - 1);
        String seletectedColumnNumsStr = seletectedColumnNums.toString();
        seletectedColumnNumsStr = seletectedColumnNumsStr.substring(1, seletectedColumnNumsStr.length() - 1);
        String delimiterStr  = String.valueOf(delimiter);
        if((delimiter ^ '|') * (delimiter ^ '&') * (delimiter ^ '>') * (delimiter ^ '<') == 0) {
            delimiterStr = "\\" + delimiter;
        }

        commands.add("python");
        commands.add("train.py");
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
        commands.add("-hiddenlayers");
        commands.add(String.valueOf(hiddenLayers));
        commands.add("-inputdaatapath");
        commands.add(inputDataPath);
        commands.add("-seletectedcolumnnums");
        commands.add(seletectedColumnNumsStr);
        commands.add("-alg");
        commands.add(alg);
        commands.add("-delimiter");
        commands.add(delimiterStr);
        commands.add("-lossfunc");
        commands.add(lossFunc);
        commands.add("-optimizer");
        commands.add(optimizer);

        return commands;
    }

}
