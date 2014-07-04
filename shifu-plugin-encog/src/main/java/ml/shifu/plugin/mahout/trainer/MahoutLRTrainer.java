package ml.shifu.plugin.mahout.trainer;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import ml.shifu.core.container.NNParams;
import ml.shifu.core.container.PMMLDataSet;
import ml.shifu.core.di.spi.Trainer;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;

import org.apache.mahout.classifier.sgd.ElasticBandPrior;
import org.apache.mahout.classifier.sgd.L1;
import org.apache.mahout.classifier.sgd.L2;
import org.apache.mahout.classifier.sgd.OnlineLogisticRegression;
import org.apache.mahout.classifier.sgd.PriorFunction;
import org.apache.mahout.classifier.sgd.TPrior;
import org.apache.mahout.classifier.sgd.UniformPrior;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.neural.networks.layers.BasicLayer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.fasterxml.jackson.databind.ObjectMapper;

public class MahoutLRTrainer implements Trainer {

    public static final String PRIORFUNCTION = "PriorFunction";
    public static final String DECISIONFOREST = "DecisionForest";
    public static final String NUM_HIDDEN_NODES = "NumHiddenNodes";
    private static Logger log = LoggerFactory.getLogger(MahoutLRTrainer.class);
    private static final DecimalFormat df = new DecimalFormat("0.000000");
    private List<MahoutDataPair> fullDataSet = new ArrayList<MahoutDataPair>();
    private OnlineLogisticRegression lrModel;
    private static Map<String, PriorFunction> priorFunctionMap = new HashMap<String, PriorFunction>() {
        {
            put("ElasticBandPrior", new ElasticBandPrior());
            put("L1", new L1());
            put("L2", new L2());
            put("UniformPrior", new UniformPrior());
        }
    };

    public void train(PMMLDataSet dataSet, Params rawParams) throws Exception {
        NNParams params = parseParams(rawParams);
        String trainerID = rawParams.get("trainerID").toString();
        String pathOutput = rawParams.get("pathOutput").toString();
        File outputFolder = new File(pathOutput);
        if (!outputFolder.exists()) {
            outputFolder.mkdirs();
        }
        Integer numActiveFields = PMMLUtils.getNumActiveMiningFields(dataSet
                .getMiningSchema());
        Integer numTargetFields = PMMLUtils.getNumTargetMiningFields(dataSet
                .getMiningSchema());
        // prepare data set
        convertDataSet(dataSet, numActiveFields, numTargetFields);
        splitDataSet(params.getSplitRatio());
        // create neural network
        OnlineLogisticRegression network = createLRModel(rawParams,
                numActiveFields);
        // train the data
        for (MahoutDataPair input : fullDataSet) {
            if (!input.isEvalData)
                network.train((int) input.getIdealData()[0],
                        input.getMahoutEvalVector());
        }
        // save neural network
        String path = pathOutput + "/model_" + trainerID + "_" + 1;
        saveMLModel(path);
        // evaluate and calculate errors
        String extra = " <-- NN saved: " + path;
        log.info("  Trainer-" + trainerID + "\n Train Error: "
                + df.format(getTestSetError()) + "\n" + extra);
        log.info("Trainer #" + trainerID + " is Finished!");
    }

    private void convertDataSet(PMMLDataSet pmmlDataSet, int numActiveFields,
            int numTargetFields) {
        List<MiningField> miningFields = pmmlDataSet.getMiningSchema()
                .getMiningFields();
        Integer numFields = miningFields.size();
        for (List<Object> row : pmmlDataSet.getRows()) {
            if (numFields != row.size()) {
                throw new RuntimeException(
                        "MiningSchema does not match data: Number of MiningFields = "
                                + numFields + ", Number of data fields = "
                                + row.size());
            }
            double[] input = new double[numActiveFields];
            double[] ideal = new double[numTargetFields];

            int inputPtr = 0;
            int idealPtr = 0;
            for (int i = 0; i < numFields; i++) {
                if (miningFields.get(i).getUsageType()
                        .equals(FieldUsageType.ACTIVE)) {
                    input[inputPtr] = Double.valueOf(row.get(i).toString());
                    inputPtr += 1;
                } else if (miningFields.get(i).getUsageType()
                        .equals(FieldUsageType.TARGET)) {
                    ideal[idealPtr] = Double.valueOf(row.get(i).toString());
                    idealPtr += 1;
                }
            }
            fullDataSet.add(new MahoutDataPair(input, ideal));
        }
    }

    private void splitDataSet(Double splitRatio) {
        Random random = new Random();
        for (MahoutDataPair pair : fullDataSet) {
            if (random.nextDouble() <= splitRatio) {
                pair.setEvalData(true);
            }
        }
    }

    private Double calculateMSE(OnlineLogisticRegression network) {
        double mseError = 0;
        long numRecords = fullDataSet.size();
        for (MahoutDataPair pair : fullDataSet) {
            double predict = network.classifyScalar(pair.getMahoutEvalVector());
            double idealData = pair.getIdealData()[0];
            mseError += Math.pow(idealData - predict, 2.0);
        }
        return mseError / numRecords;
    }

    private NNParams parseParams(Params rawParams) throws Exception {
        ObjectMapper jsonMapper = new ObjectMapper();
        String jsonString = jsonMapper.writeValueAsString(rawParams);
        return jsonMapper.readValue(jsonString, NNParams.class);
    }

    private OnlineLogisticRegression createLRModel(Params params,
            int inputSize) {
        String priorFunction = params.get(PRIORFUNCTION).toString();
        for (Map.Entry<String, PriorFunction> entry : priorFunctionMap
                .entrySet()) {
            if (priorFunction.equalsIgnoreCase(entry.getKey()))  {
                lrModel = new OnlineLogisticRegression(2, inputSize, entry.getValue());
                return lrModel;
            }
        }
        if (priorFunction.equalsIgnoreCase("TPrior")) {
            double df = Double.parseDouble(params.get(DECISIONFOREST).toString());
            lrModel = new OnlineLogisticRegression(2, inputSize, new TPrior(df));
            return lrModel;
        }
        lrModel = new OnlineLogisticRegression(2, inputSize, new L1());
        return lrModel;
    }

    private double getTestSetError() {
        return calculateMSE(this.lrModel);
        // return calculateMSEParallel(this.network, this.validSet);
    }

    private void saveMLModel(String path) throws IOException {

        // EncogDirectoryPersistence.saveObject(new File(path), network);
    }

}
