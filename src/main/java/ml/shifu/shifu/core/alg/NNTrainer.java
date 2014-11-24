/**
 * Copyright [2012-2014] eBay Software Foundation
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

import ml.shifu.shifu.container.ModelInitInputObject;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.AbstractTrainer;
import ml.shifu.shifu.core.MSEWorker;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.JSONUtils;
import org.encog.engine.network.activation.*;
import org.encog.mathutil.IntRange;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.neural.networks.training.propagation.scg.ScaledConjugateGradient;
import org.encog.persist.EncogDirectoryPersistence;
import org.encog.util.concurrency.DetermineWorkload;
import org.encog.util.concurrency.EngineConcurrency;
import org.encog.util.concurrency.TaskGroup;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.util.*;

/**
 * Neural network trainer
 */
public class NNTrainer extends AbstractTrainer {

    public static final String NUM_HIDDEN_LAYERS = "NumHiddenLayers";
    public static final String ACTIVATION_FUNC = "ActivationFunc";
    public static final String NUM_HIDDEN_NODES = "NumHiddenNodes";
    public static final String LEARNING_RATE = "LearningRate";
    public static final String PROPAGATION = "Propagation";

    private static Logger log = LoggerFactory.getLogger(NNTrainer.class);
    private final static double Epsilon = 1.0;  // set the weight range in [-INIT_EPSILON INIT_EPSILON];

    public static final Map<String, Double> defaultLearningRate;
    public static final Map<String, String> learningAlgMap;

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

    public NNTrainer(ModelConfig modelConfig, int trainerID, Boolean dryRun) {
        super(modelConfig, trainerID, dryRun);
    }


    @SuppressWarnings("unchecked")
    public void buildNetwork() {
        network = new BasicNetwork();

        network.addLayer(new BasicLayer(new ActivationLinear(), true, trainSet.getInputSize()));

        int numLayers = (Integer) modelConfig.getParams().get(NUM_HIDDEN_LAYERS);
        List<String> actFunc = (List<String>) modelConfig.getParams().get(ACTIVATION_FUNC);
        List<Integer> hiddenNodeList = (List<Integer>) modelConfig.getParams().get(NUM_HIDDEN_NODES);

        if (numLayers != 0 && (numLayers != actFunc.size() || numLayers != hiddenNodeList.size())) {
            throw new RuntimeException("the number of layer do not equal to the number of activation function or the function list and node list empty");
        }

        log.info("    - total " + numLayers + " layers, each layers are: " + Arrays.toString(hiddenNodeList.toArray()) + " the activation function are: " + Arrays.toString(actFunc.toArray()));

        for (int i = 0; i < numLayers; i++) {
            String func = actFunc.get(i);
            Integer numHiddenNode = hiddenNodeList.get(i);
            //java 6
            if ("linear".equalsIgnoreCase(func)) {
                network.addLayer(new BasicLayer(new ActivationLinear(), true, numHiddenNode));
            } else if (func.equalsIgnoreCase("sigmoid")) {
                network.addLayer(new BasicLayer(new ActivationSigmoid(), true, numHiddenNode));
            } else if (func.equalsIgnoreCase("tanh")) {
                network.addLayer(new BasicLayer(new ActivationTANH(), true, numHiddenNode));
            } else if (func.equalsIgnoreCase("log")) {
                network.addLayer(new BasicLayer(new ActivationLOG(), true, numHiddenNode));
            } else if (func.equalsIgnoreCase("sin")) {
                network.addLayer(new BasicLayer(new ActivationSIN(), true, numHiddenNode));
            } else {
                log.info("Unsupported activation function: " + func + " !! Set this layer activation function to be Sigmoid ");
                network.addLayer(new BasicLayer(new ActivationSigmoid(), true, numHiddenNode));
            }
        }

        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, trainSet.getIdealSize()));
        network.getStructure().finalizeStructure();
        if (!modelConfig.isFixInitialInput()) {
            network.reset();
        } else {
            int numWeight = 0;
            for (int i = 0; i < network.getLayerCount() - 1; i++) {
                numWeight = numWeight + network.getLayerTotalNeuronCount(i) * network.getLayerNeuronCount(i + 1);
            }
            log.info("    - You have " + numWeight + " weights to be initialize");
            loadWeightsInput(numWeight);
        }
    }

    @Override
    public void train() throws IOException {
        log.info("Using neural network algorithm...");

        if (this.dryRun == true) {
            log.info("Start Training(Dry Run)... Model #" + this.trainerID);
        } else {
            log.info("Start Training... Model #" + this.trainerID);
        }

        log.info("    - Input Size: " + trainSet.getInputSize());
        log.info("    - Ideal Size: " + trainSet.getIdealSize());

        //set up the model
        buildNetwork();

        Propagation mlTrain = getMLTrain();
        mlTrain.setThreadCount(0);

        if (this.dryRun == true) {
            return;
        }

        int epochs = this.modelConfig.getNumTrainEpochs();
        int factor = Math.max(epochs / 50, 10);

        setBaseMSE(Double.MAX_VALUE);

        for (int i = 0; i < epochs; i++) {
            mlTrain.iteration();

            if (i % factor == 0) {
                this.saveTmpNN(i);
            }

            double validMSE = (this.validSet.getRecordCount() > 0) ? getValidSetError() : mlTrain.getError();

            String extra = "";
            if (validMSE < getBaseMSE()) {
                setBaseMSE(validMSE);
                saveNN();
                extra = " <-- NN saved: ./models/model" + this.trainerID + ".nn";
            }

            log.info("  Trainer-" + trainerID + "> Epoch #" + (i + 1)
                    + " Train Error: " + df.format(mlTrain.getError())
                    + " Validation Error: " + ((this.validSet.getRecordCount() > 0) ? df.format(validMSE) : "N/A") + " " + extra);

        }

        mlTrain.finishTraining();
        log.info("Trainer #" + this.trainerID + " is Finished!");
    }

    public BasicNetwork getNetwork() {
        return network;
    }


    /**
     * @param network the network to set
     */
    public void setNetwork(BasicNetwork network) {
        this.network = network;
    }

    private Propagation getMLTrain() {
        //String alg = this.modelConfig.getLearningAlgorithm();
        String alg = (String) modelConfig.getParams().get(PROPAGATION);
        if (!(defaultLearningRate.containsKey(alg))) {
            throw new RuntimeException("Leanring Algorithm is not valid: " + alg);
        }

        //Double rate = this.modelConfig.getLearningRate();
        Double rate = defaultLearningRate.get(alg);
        Object rateObj = modelConfig.getParams().get(LEARNING_RATE);
        if (rateObj instanceof Double) {
            rate = (Double) rateObj;
        } else if (rateObj instanceof Integer) {
            // change like this, because user may set it as integer
            rate = Double.valueOf(((Integer) rateObj).doubleValue());
        } else if (rateObj instanceof Float) {
            rate = Double.valueOf(((Float) rateObj).doubleValue());
        }

        log.info("    - Learning Algorithm: " + learningAlgMap.get(alg));
        if (alg.equals("Q") || alg.equals("B") || alg.equals("M")) {
            log.info("    - Learning Rate: " + rate);
        }

        if (alg.equals("B")) {
            return new Backpropagation(network, trainSet, rate, 0);
        } else if (alg.equals("Q")) {
            return new QuickPropagation(network, trainSet, rate);
        } else if (alg.equals("M")) {
            return new ManhattanPropagation(network, trainSet, rate);
        } else if (alg.equals("R")) {
            return new ResilientPropagation(network, trainSet);
        } else if (alg.equals("S")) {
            return new ScaledConjugateGradient(network, trainSet);
        } else {
            return null;
        }
    }

    private double getValidSetError() {
        //return calculateMSE(this.network, this.validSet);
        return calculateMSEParallel(this.network, this.validSet);
    }

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
            workers[index++] = new MSEWorker((BasicNetwork)
                    network.clone(), this,
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

    private void saveNN() throws IOException {
        File folder = new File(pathFinder.getModelsPath(SourceType.LOCAL));
        if (!folder.exists()) {
            folder.mkdirs();
        }
        EncogDirectoryPersistence.saveObject(new File(folder, "model" + this.trainerID + ".nn"), network);
    }

    private void saveTmpNN(int epoch) throws IOException {
        File tmpFolder = new File(pathFinder.getTmpModelsPath(SourceType.LOCAL));
        if (!tmpFolder.exists()) {
            tmpFolder.mkdirs();
        }

        EncogDirectoryPersistence.saveObject(new File(tmpFolder, "model" + trainerID + "-" + epoch + ".nn"), network);
    }

    public MLDataSet getValidSet() {
        return validSet;
    }

    public Double getBaseMSE() {
        if (baseMSE == null) {
            log.error("baseMSE is not available. Run train() First!");
            return null;
        }
        return baseMSE;
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
                BufferedReader reader = ShifuFileUtils.getReader("./init" + this.trainerID + ".json", SourceType.LOCAL);
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
    }
}