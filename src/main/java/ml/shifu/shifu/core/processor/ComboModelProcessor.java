package ml.shifu.shifu.core.processor;

import ml.shifu.shifu.container.obj.ComboModelTrain;
import ml.shifu.shifu.container.obj.ModelTrainConf;
import ml.shifu.shifu.container.obj.VarTrainConf;
import ml.shifu.shifu.core.validator.ModelInspector;
import ml.shifu.shifu.util.JSONUtils;
import org.apache.avro.generic.GenericData;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by zhanhu on 12/5/16.
 */
public class ComboModelProcessor extends BasicModelProcessor implements Processor {

    private static Logger LOG = LoggerFactory.getLogger(ComboModelProcessor.class);

    public static enum ComboStep {
        NEW, INIT, RUN, EVAL
    }
    public static final String ALG_DELIMITER = ",";

    private ComboStep comboStep;
    private String algorithms;

    private List<ModelTrainConf.ALGORITHM> comboAlgs;

    public ComboModelProcessor(ComboStep comboStep) {
        this.comboStep = comboStep;
    }

    public ComboModelProcessor(ComboStep comboStep, String algorithms) {
        this(comboStep);
        this.algorithms = algorithms;
    }

    @Override
    public int run() throws Exception {
        LOG.info("Start to run combo, step - {}", this.comboStep);

        int status = 0;

        setUp(ModelInspector.ModelStep.COMBO);

        switch (comboStep) {
            case NEW:
                status = createNewCombo();
                break;
        }

        return status;
    }

    private int createNewCombo() {
        int status = validate(algorithms);
        if ( status > 0 ) {
            LOG.error("Fail to validate combo algorithms - {}.", algorithms);
            return status;
        }

        ComboModelTrain comboModelTrain = new ComboModelTrain();
        comboModelTrain.setUidColumnName("");

        List<VarTrainConf> varTrainConfList = new ArrayList<VarTrainConf>(this.comboAlgs.size() - 1);
        for ( int i = 0; i < this.comboAlgs.size() - 1; i ++ ) {
            varTrainConfList.add(createVarTrainConf(this.comboAlgs.get(i)));
        }
        comboModelTrain.setVarTrainConfList(varTrainConfList);

        comboModelTrain.setFusionModelTrainConf(createModelTrainConf(this.comboAlgs.get(this.comboAlgs.size() - 1)));

        status = saveComboTrain(comboModelTrain);
        
        return status;
    }

    private int validate(String algorithms) {
        if (StringUtils.isBlank(algorithms) ) {
            LOG.error("The combo algorithms should not be empty");
            return 1;
        }

        String[] algs = algorithms.split(ALG_DELIMITER);
        if ( algs.length < 3 ) {
            LOG.error("At least, you should have 2 basic algorithms, and 1 assembling algorithm.");
            return 2;
        }

        this.comboAlgs = new ArrayList<ModelTrainConf.ALGORITHM>();
        for ( String alg : algs ) {
            try {
                ModelTrainConf.ALGORITHM algorithm = ModelTrainConf.ALGORITHM.valueOf(alg);
                if ( algorithm == null ) {
                    LOG.error("Unsupported algorithm - {}", alg);
                    return 3;
                }
                this.comboAlgs.add(algorithm);
            } catch (Throwable t) {
                LOG.error("Unsupported algorithm - {}", alg);
                return 3;
            }
        }
        return 0;
    }

    private VarTrainConf createVarTrainConf(ModelTrainConf.ALGORITHM alg) {
        VarTrainConf varTrainConf = new VarTrainConf();
        varTrainConf.setVariables(new ArrayList<String>());
        varTrainConf.setModelTrainConf(createModelTrainConf(alg));
        return varTrainConf;
    }

    private ModelTrainConf createModelTrainConf(ModelTrainConf.ALGORITHM alg) {
        ModelTrainConf trainConf = new ModelTrainConf();

        trainConf.setAlgorithm(alg.name());
        trainConf.setEpochsPerIteration(1);
        trainConf.setParams(ModelTrainConf.createParamsByAlg(alg, trainConf));
        trainConf.setNumTrainEpochs(100);
        if(ModelTrainConf.ALGORITHM.NN.equals(alg)) {
            trainConf.setNumTrainEpochs(200);
        } else if(ModelTrainConf.ALGORITHM.SVM.equals(alg)) {
            trainConf.setNumTrainEpochs(100);
        } else if(ModelTrainConf.ALGORITHM.RF.equals(alg)) {
            trainConf.setNumTrainEpochs(20000);
        } else if(ModelTrainConf.ALGORITHM.GBT.equals(alg)) {
            trainConf.setNumTrainEpochs(20000);
        } else if(ModelTrainConf.ALGORITHM.LR.equals(alg)) {
            trainConf.setNumTrainEpochs(100);
        }
        trainConf.setBaggingWithReplacement(true);

        return trainConf;
    }


    private int saveComboTrain(ComboModelTrain comboModelTrain) {
        try {
            JSONUtils.writeValue(new File("ComboTrain.json"), comboModelTrain);
        } catch (Exception e) {
            LOG.error("Fail to save ComboModelTrain object to ComboTrain.json");
            return 1;
        }
        return 0;
    }

}
