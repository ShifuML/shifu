package ml.shifu.plugin.encog.trainer;

import com.fasterxml.jackson.databind.ObjectMapper;
import ml.shifu.core.container.HiddenLayer;
import ml.shifu.core.container.NNParams;
import ml.shifu.core.container.PMMLDataSet;
import ml.shifu.core.di.spi.Trainer;
import ml.shifu.core.util.PMMLUtils;
import ml.shifu.core.util.Params;
import ml.shifu.plugin.encog.adapter.EncogNeuralNetworkToPMMLAdapter;
import org.dmg.pmml.FieldUsageType;
import org.dmg.pmml.MiningField;
import org.dmg.pmml.Model;
import org.encog.engine.network.activation.*;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.propagation.scg.ScaledConjugateGradient;
import org.encog.persist.EncogDirectoryPersistence;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.*;

public class EncogNNTrainer implements Trainer {

    public static final String NUM_HIDDEN_LAYERS = "NumHiddenLayers";
    public static final String ACTIVATION_FUNC = "ActivationFunc";
    public static final String NUM_HIDDEN_NODES = "NumHiddenNodes";
    public static final String LEARNING_RATE = "LearningRate";
    public static final String PROPAGATION = "Propagation";

    private static Logger log = LoggerFactory.getLogger(EncogNNTrainer.class);
    private final static double Epsilon = 1.0;  // set the weight range in [-INIT_EPSILON INIT_EPSILON];

    private static final Map<String, Double> defaultLearningRate;
    private static final Map<String, String> learningAlgMap;

    private static final DecimalFormat df = new DecimalFormat("0.000000");


    private MLDataSet trainDataSet;
    private MLDataSet testDataSet;

    private Double minError;


    private BasicNetwork network;

    static {
        defaultLearningRate = new HashMap<String, Double>();
        defaultLearningRate.put("S", 0.1);
        defaultLearningRate.put("R", 0.1);
        defaultLearningRate.put("Q", 2.0);
        defaultLearningRate.put("B", 0.01);
        defaultLearningRate.put("M", 0.00001);

        learningAlgMap = new HashMap<String, String>();
        learningAlgMap.put("S", "Scaled Conjugate Gradient");
        learningAlgMap.put("R", "Resilient Propagation");
        learningAlgMap.put("M", "Manhattan Propagation");
        learningAlgMap.put("B", "Back Propagation");
        learningAlgMap.put("Q", "Quick Propagation");
    }




    public void train(Model pmmlModel, PMMLDataSet dataSet, Params rawParams) throws Exception {



        //String trainerID = (String) params.get("trainerID", "Trainer");


        //log.info("Using neural network algorithm...");

        //log.info("Start Training... Model #" + trainerID);

        //Double splitRatio = (Double) params.get("splitRatio", 0.8);

        NNParams params = parseParams(rawParams);

        String trainerID = rawParams.get("trainerID").toString();
        String pathOutput = rawParams.get("pathOutput").toString();

        File outputFolder = new File(pathOutput);

        if (!outputFolder.exists()) {
            outputFolder.mkdirs();
        }


        MLDataSet fullDataSet = convertDataSet(dataSet);

        trainDataSet = new BasicMLDataSet();
        testDataSet = new BasicMLDataSet();

        splitDataSet(fullDataSet, params.getSplitRatio(), trainDataSet, testDataSet);




        log.info("    - Input Size: " + trainDataSet.getInputSize());
        log.info("    - Ideal Size: " + trainDataSet.getIdealSize());

        log.info("    - TrainSet: " + trainDataSet.getRecordCount());
        log.info("    - TestSet: " + testDataSet.getRecordCount());


        BasicNetwork network = createNetwork(params);
        Propagation mlTrain = getMLTrain(network, trainDataSet, params);
        mlTrain.setThreadCount(0);

        int epochs = params.getNumEpochs();
        int factor = Math.max(epochs / 50, 10);

        minError = Double.MAX_VALUE;

        EncogNeuralNetworkToPMMLAdapter adapter = new EncogNeuralNetworkToPMMLAdapter();

        for (int i = 0; i < epochs; i++) {
            mlTrain.iteration();

            double testError = (testDataSet.getRecordCount() > 0) ? getTestSetError() : mlTrain.getError();

            String extra = "";
            if (testError < minError) {
                minError = testError;

                String path = pathOutput + "/model_" + trainerID + "_" + (i + 1);

                saveNN(path);
                extra = " <-- NN saved: " + path;
            }

            log.info("  Trainer-" + trainerID + "> Epoch #" + (i + 1)
                    + " Train Error: " + df.format(mlTrain.getError())
                    + " Test Error: " + ((testDataSet.getRecordCount() > 0) ? df.format(testError) : "N/A") + " " + extra);

        }

        mlTrain.finishTraining();

        adapter.exec(network, pmmlModel);

        //Model model = adapter.exec(network, pmmlModel);



        log.info("Trainer #" + trainerID + " is Finished!");
    }


    public MLDataSet convertDataSet(PMMLDataSet pmmlDataSet) {
        MLDataSet convertedDataSet = new BasicMLDataSet();

        List<MiningField> miningFields = pmmlDataSet.getMiningSchema().getMiningFields();
        Integer numFields = miningFields.size();
        Integer numActiveFields = PMMLUtils.getNumActiveMiningFields(pmmlDataSet.getMiningSchema());
        Integer numTargetFields = PMMLUtils.getNumTargetMiningFields(pmmlDataSet.getMiningSchema());

        for (List<Object> row : pmmlDataSet.getRows()) {

            if (numFields != row.size()) {
                throw new RuntimeException("MiningSchema does not match data: Number of MiningFields = " + numFields + ", Number of data fields = " + row.size());
            }

            double[] input = new double[numActiveFields];
            double[] ideal = new double[numTargetFields];

            int inputPtr = 0;
            int idealPtr = 0;

            for (int i = 0; i < numFields; i++) {
                if (miningFields.get(i).getUsageType().equals(FieldUsageType.ACTIVE)) {
                    input[inputPtr] = Double.valueOf(row.get(i).toString());
                    inputPtr += 1;
                } else if (miningFields.get(i).getUsageType().equals(FieldUsageType.TARGET)) {
                    ideal[idealPtr] = Double.valueOf(row.get(i).toString());
                    idealPtr += 1;
                }
            }

            MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));

            convertedDataSet.add(pair);
        }

        return convertedDataSet;
    }

    public void splitDataSet(MLDataSet fullDataSet, Double splitRatio, MLDataSet trainDataSet, MLDataSet testDataSet) {



        Random random = new Random();

        for (MLDataPair pair : fullDataSet) {
            if (random.nextDouble() <= splitRatio) {
                trainDataSet.add(pair);
            } else {
                testDataSet.add(pair);
            }
        }
    }
    private NNParams parseParams(Params rawParams) throws Exception {
        ObjectMapper jsonMapper = new ObjectMapper();
        String jsonString = jsonMapper.writeValueAsString(rawParams);
        return jsonMapper.readValue(jsonString, NNParams.class);
    }



   private BasicNetwork createNetwork(NNParams params) {
        network = new BasicNetwork();

        network.addLayer(new BasicLayer(new ActivationLinear(), true, trainDataSet.getInputSize()));

        int numLayers = params.getHiddenLayers().size();
        //List<String> actFunc = (List<String>) params.get(ACTIVATION_FUNC);
        //List<Integer> hiddenNodeList = (List<Integer>) params.get(NUM_HIDDEN_NODES);

        //if (numLayers != 0 && (numLayers != actFunc.size() || numLayers != hiddenNodeList.size())) {
        //    throw new RuntimeException("the number of layer do not equal to the number of activation function or the function list and node list empty");
        //}

        //log.info("    - total " + numLayers + " layers, each layers are: " + Arrays.toString(hiddenNodeList.toArray()) + " the activation function are: " + Arrays.toString(actFunc.toArray()));



        for (HiddenLayer hiddenLayer : params.getHiddenLayers()) {

            String activationFunction = hiddenLayer.getActivationFunction();
            int numHiddenNodes = hiddenLayer.getNumHiddenNodes();
            if (activationFunction.equalsIgnoreCase("linear")) {
                network.addLayer(new BasicLayer(new ActivationLinear(), true, numHiddenNodes));
            } else if (activationFunction.equalsIgnoreCase("sigmoid")) {
                network.addLayer(new BasicLayer(new ActivationSigmoid(), true, numHiddenNodes));
            } else if (activationFunction.equalsIgnoreCase("tanh")) {
                network.addLayer(new BasicLayer(new ActivationTANH(), true, numHiddenNodes));
            } else if (activationFunction.equalsIgnoreCase("log")) {
                network.addLayer(new BasicLayer(new ActivationLOG(), true, numHiddenNodes));
            } else if (activationFunction.equalsIgnoreCase("sin")) {
                network.addLayer(new BasicLayer(new ActivationSIN(), true, numHiddenNodes));
            } else {
                throw new RuntimeException("Unsupported ActivationFunction: " + activationFunction);
            }
        }

        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, trainDataSet.getIdealSize()));
        network.getStructure().finalizeStructure();
        network.reset();
        /*if (!modelConfig.isFixInitialInput()) {
            network.reset();
        } else {
            int numWeight = 0;
            for (int i = 0; i < network.getLayerCount() - 1; i++) {
                numWeight = numWeight + network.getLayerTotalNeuronCount(i) * network.getLayerNeuronCount(i + 1);
            }
            log.info("    - You have " + numWeight + " weights to be initialize");
            loadWeightsInput(numWeight);
        } */

       return network;
    }

    private Propagation getMLTrain(BasicNetwork network, MLDataSet trainSet, NNParams params) {
        //String alg = this.modelConfig.getLearningAlgorithm();
        String algorithm = params.getAlgorithm();
        if (!(defaultLearningRate.containsKey(algorithm))) {
            throw new RuntimeException("Leanring Algorithm is not valid: " + algorithm);
        }

        //Double rate = this.modelConfig.getLearningRate();

        Double rate = params.getLearningRate();
        if (rate == null) {
            rate = defaultLearningRate.get(algorithm);
        }

        log.info("    - Learning Algorithm: " + learningAlgMap.get(algorithm));
        if (algorithm.equals("Q") || algorithm.equals("B") || algorithm.equals("M")) {
            log.info("    - Learning Rate: " + rate);
        }

        if (algorithm.equals("B")) {
            return new Backpropagation(network, trainSet, rate, 0);
        } else if (algorithm.equals("Q")) {
            return new QuickPropagation(network, trainSet, rate);
        } else if (algorithm.equals("M")) {
            return new ManhattanPropagation(network, trainSet, rate);
        } else if (algorithm.equals("R")) {
            return new ResilientPropagation(network, trainSet);
        } else if (algorithm.equals("S")) {
            return new ScaledConjugateGradient(network, trainSet);
        } else {
            return null;
        }
    }
      /*

    public BasicNetwork getNetwork() {
        return network;
    }


    /**
     * @param network the network to set
     */
    /*
    public void setNetwork(BasicNetwork network) {
        this.network = network;
    }


    */
    private double getTestSetError() {
        return calculateMSE(this.network, testDataSet);
        //return calculateMSEParallel(this.network, this.validSet);
    }

    public static Double calculateMSE(BasicNetwork network, MLDataSet dataSet) {

        double mse = 0;
        long numRecords = dataSet.getRecordCount();
        for (int i = 0; i < numRecords; i++) {

            // Encog 3.1
            // MLDataPair pair = dataSet.get(i);

            // Encog 3.0
            double[] input = new double[dataSet.getInputSize()];
            double[] ideal = new double[1];
            MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));

            dataSet.getRecord(i, pair);

            MLData result = network.compute(pair.getInput());

            double tmp = result.getData()[0] - pair.getIdeal().getData()[0];
            mse += tmp * tmp;
        }
        mse = mse / numRecords;

        return mse;
    }

    /*

    public double calculateMSEParallel(BasicNetwork network, MLDataSet dataSet) {

        totalError = 0;

        int numRecords = (int) dataSet.getRecordCount();

        assert numRecords > 0;
        // setup workers
        final DetermineWorkload determine = new DetermineWorkload(0, numRecords);

        // nice little workaround
        MSEWorker[] workers = new MSEWorker[determine.getThreadCount()];

        int index = 0;
        TaskGroup group = EngineConcurrency.getInstance().createTaskGroup();
        for (final IntRange r : determine.calculateWorkers()) {
            workers[index++] = new MSEWorker((BasicNetwork) network.clone(), this,
                    dataSet.openAdditional(), r.getLow(), r.getHigh()
            );
        }

        for (final MSEWorker worker : workers) {
            EngineConcurrency.getInstance().processTask(worker, group);
        }

        group.waitForComplete();

        double mse = totalError / numRecords;

        return mse;
    }
    */


    private void saveNN(String path) throws IOException {
        //File folder = new File();
        //if (!folder.exists()) {
        //    folder.mkdirs();
        //}

        EncogDirectoryPersistence.saveObject(new File(path), network);
    }
     /*
    private void saveTmpNN(int epoch) throws IOException {
        File tmpFolder = new File(pathFinder.getTmpModelsPath(RawSourceData.SourceType.LOCAL));
        if (!tmpFolder.exists()) {
            tmpFolder.mkdirs();
        }

        EncogDirectoryPersistence.saveObject(new File(tmpFolder, "model" + trainerID + "-" + epoch + ".nn"), network);
    }

    public MLDataSet getValidSet() {
        return validSet;
    }

    public Double getMinError() {
        if (minError == null) {
            log.error("minError is not available. Run train() First!");
            return null;
        }
        return minError;
    }

    private void loadWeightsInput(int numWeights) {
        try {
            File file = new File("./init" + this.trainerID + ".json");
            if (!file.exists()) {
                file.createNewFile();

                ModelInitInputObject io = new ModelInitInputObject();
                io.setWeights(randomSetWeights(numWeights));
                io.setNumWeights(numWeights);

                setWeights(io.getWeights());

                JSONUtils.writeValue(file, io);
            } else {
                BufferedReader reader = ShifuFileUtils.getReader("./init" + this.trainerID + ".json", RawSourceData.SourceType.LOCAL);
                ModelInitInputObject io = JSONUtils.readValue(reader, ModelInitInputObject.class);
                if (io == null) {
                    io = new ModelInitInputObject();
                }
                if (io.getNumWeights() != numWeights) {

                    io.setNumWeights(numWeights);
                    io.setWeights(randomSetWeights(numWeights));

                    file.createNewFile();
                    JSONUtils.writeValue(file, io);
                }

                setWeights(io.getWeights());
                reader.close();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void setWeights(List<Double> weights) {
        if (network == null) return;
        int i = 0;
        for (int numLayer = 0; numLayer < network.getLayerCount() - 1; numLayer++) {

            int fromCount = network.getLayerTotalNeuronCount(numLayer);
            int toCount = network.getLayerNeuronCount(numLayer + 1);
            for (int fromNeuron = 0; fromNeuron < fromCount; fromNeuron++) {
                for (int toNeuron = 0; toNeuron < toCount; toNeuron++) {
                    network.setWeight(numLayer, fromNeuron, toNeuron, weights.get(i++));
                }
            }
        }
    }

    private List<Double> randomSetWeights(int numWeights) {
        List<Double> weights = new ArrayList<Double>();
        for (int i = 0; i < numWeights; i++) {
            weights.add(this.random.nextDouble() * 2 * Epsilon - Epsilon);
        }
        return weights;
    }                 */



}
