/*
 * Copyright [2012-2014] PayPal Software Foundation
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
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.ModelInitInputObject;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.AbstractTrainer;
import ml.shifu.shifu.core.ConvergeJudger;
import ml.shifu.shifu.core.MSEWorker;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.JSONUtils;

import org.apache.commons.io.FileUtils;
import org.encog.engine.network.activation.ActivationLOG;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSIN;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.engine.network.activation.ActivationTANH;
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

/**
 * Neural network trainer
 */
public class NNTrainer extends AbstractTrainer {

    private static final Logger LOG = LoggerFactory.getLogger(NNTrainer.class);
    private final static double Epsilon = 1.0; // set the weight range in [-INIT_EPSILON INIT_EPSILON];

    public static final Map<String, Double> defaultLearningRate;
    public static final Map<String, String> learningAlgMap;

    private BasicNetwork network;
    private volatile boolean toPersistentModel = true;
    private volatile boolean toLoggingProcess = true;

    /**
     * Convergence judger instance for convergence criteria checking.
     */
    private ConvergeJudger judger = new ConvergeJudger();

    static {
        // TODO use UnmodifiableMap or use other immutable Collections such as guava's
        Map<String, Double> tmpLearningRate = new HashMap<String, Double>();
        tmpLearningRate.put("S", 0.1);
        tmpLearningRate.put("R", 0.1);
        tmpLearningRate.put("Q", 2.0);
        tmpLearningRate.put("B", 0.01);
        tmpLearningRate.put("M", 0.00001);
        defaultLearningRate = Collections.unmodifiableMap(tmpLearningRate);

        Map<String, String> tmpLearningAlgMap = new HashMap<String, String>();
        tmpLearningAlgMap.put("S", "Scaled Conjugate Gradient");
        tmpLearningAlgMap.put("R", "Resilient Propagation");
        tmpLearningAlgMap.put("M", "Manhattan Propagation");
        tmpLearningAlgMap.put("B", "Back Propagation");
        tmpLearningAlgMap.put("Q", "Quick Propagation");
        learningAlgMap = Collections.unmodifiableMap(tmpLearningAlgMap);
    }

    public NNTrainer(ModelConfig modelConfig, int trainerID, Boolean dryRun) {
        super(modelConfig, trainerID, dryRun);
    }

    @SuppressWarnings("unchecked")
    public void buildNetwork() {
        network = new BasicNetwork();

        network.addLayer(new BasicLayer(new ActivationLinear(), true, trainSet.getInputSize()));

        int numLayers = (Integer) modelConfig.getParams().get(CommonConstants.NUM_HIDDEN_LAYERS);
        List<String> actFunc = (List<String>) modelConfig.getParams().get(CommonConstants.ACTIVATION_FUNC);
        List<Integer> hiddenNodeList = (List<Integer>) modelConfig.getParams().get(CommonConstants.NUM_HIDDEN_NODES);

        if(numLayers != 0 && (numLayers != actFunc.size() || numLayers != hiddenNodeList.size())) {
            throw new RuntimeException(
                    "the number of layer do not equal to the number of activation function or the function list and node list empty");
        }
        if(toLoggingProcess)
            LOG.info("    - total " + numLayers + " layers, each layers are: "
                    + Arrays.toString(hiddenNodeList.toArray()) + " the activation function are: "
                    + Arrays.toString(actFunc.toArray()));

        for(int i = 0; i < numLayers; i++) {
            String func = actFunc.get(i);
            Integer numHiddenNode = hiddenNodeList.get(i);
            // java 6
            if("linear".equalsIgnoreCase(func)) {
                network.addLayer(new BasicLayer(new ActivationLinear(), true, numHiddenNode));
            } else if(func.equalsIgnoreCase("sigmoid")) {
                network.addLayer(new BasicLayer(new ActivationSigmoid(), true, numHiddenNode));
            } else if(func.equalsIgnoreCase("tanh")) {
                network.addLayer(new BasicLayer(new ActivationTANH(), true, numHiddenNode));
            } else if(func.equalsIgnoreCase("log")) {
                network.addLayer(new BasicLayer(new ActivationLOG(), true, numHiddenNode));
            } else if(func.equalsIgnoreCase("sin")) {
                network.addLayer(new BasicLayer(new ActivationSIN(), true, numHiddenNode));
            } else {
                LOG.info("Unsupported activation function: " + func
                        + " !! Set this layer activation function to be Sigmoid ");
                network.addLayer(new BasicLayer(new ActivationSigmoid(), true, numHiddenNode));
            }
        }

        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, trainSet.getIdealSize()));
        network.getStructure().finalizeStructure();
        if(!modelConfig.isFixInitialInput()) {
            network.reset();
        } else {
            int numWeight = 0;
            for(int i = 0; i < network.getLayerCount() - 1; i++) {
                numWeight = numWeight + network.getLayerTotalNeuronCount(i) * network.getLayerNeuronCount(i + 1);
            }
            LOG.info("    - You have " + numWeight + " weights to be initialize");
            loadWeightsInput(numWeight);
        }
    }

    @Override
    public double train() throws IOException {
        if(toLoggingProcess)
            LOG.info("Using neural network algorithm...");

        if(toLoggingProcess) {
            if(this.dryRun) {
                LOG.info("Start Training(Dry Run)... Model #" + this.trainerID);
            } else {
                LOG.info("Start Training... Model #" + this.trainerID);
            }

            LOG.info("    - Input Size: " + trainSet.getInputSize());
            LOG.info("    - Ideal Size: " + trainSet.getIdealSize());
            LOG.info("    - Training Records Count: " + trainSet.getRecordCount());
            LOG.info("    - Validation Records Count: " + validSet.getRecordCount());
        }

        // set up the model
        buildNetwork();

        Propagation mlTrain = getMLTrain();
        mlTrain.setThreadCount(0);

        if(this.dryRun) {
            return 0.0;
        }

        int epochs = this.modelConfig.getNumTrainEpochs();
        int factor = Math.max(epochs / 50, 10);

        // Get convergence threshold from modelConfig.
        double threshold = modelConfig.getTrain().getConvergenceThreshold() == null ? 0.0 : modelConfig.getTrain()
                .getConvergenceThreshold().doubleValue();
        String formatedThreshold = df.format(threshold);

        setBaseMSE(Double.MAX_VALUE);

        for(int i = 0; i < epochs; i++) {
            mlTrain.iteration();

            if(i % factor == 0) {
                this.saveTmpNN(i);
            }

            double validMSE = (this.validSet.getRecordCount() > 0) ? getValidSetError() : mlTrain.getError();

            String extra = "";
            if(validMSE < getBaseMSE()) {
                setBaseMSE(validMSE);
                saveNN();
                extra = " <-- NN saved: ./models/model" + this.trainerID + ".nn";
            }

            if(toLoggingProcess)
                LOG.info("  Trainer-" + trainerID + "> Epoch #" + (i + 1) + " Train Error: "
                        + df.format(mlTrain.getError()) + " Validation Error: "
                        + ((this.validSet.getRecordCount() > 0) ? df.format(validMSE) : "N/A") + " " + extra);

            // Convergence judging.
            double avgErr = (mlTrain.getError() + validMSE) / 2;

            if(judger.judge(avgErr, threshold)) {
                LOG.info("Trainer-{}> Epoch #{} converged! Average Error: {}, Threshold: {}", trainerID, (i + 1),
                        df.format(avgErr), formatedThreshold);
                break;
            } else {
                if(toLoggingProcess) {
                    LOG.info("Trainer-{}> Epoch #{} Average Error: {}, Threshold: {}", trainerID, (i + 1),
                            df.format(avgErr), formatedThreshold);
                }
            }
        }

        mlTrain.finishTraining();
        if(toLoggingProcess)
            LOG.info("Trainer #" + this.trainerID + " is Finished!");
        return getBaseMSE();
    }

    public BasicNetwork getNetwork() {
        return network;
    }

    public void enableModelPersistence() {
        this.toPersistentModel = true;
    }

    public void disableModelPersistence() {
        this.toPersistentModel = false;
    }

    public void enableLogging() {
        this.toLoggingProcess = true;
    }

    public void disableLogging() {
        this.toLoggingProcess = false;
    }

    /**
     * @param network
     *            the network to set
     */
    public void setNetwork(BasicNetwork network) {
        this.network = network;
    }

    private Propagation getMLTrain() {
        // String alg = this.modelConfig.getLearningAlgorithm();
        String alg = (String) modelConfig.getParams().get(CommonConstants.PROPAGATION);
        if(!(defaultLearningRate.containsKey(alg))) {
            throw new RuntimeException("Learning algorithm is invalid: " + alg);
        }

        // Double rate = this.modelConfig.getLearningRate();
        double rate = defaultLearningRate.get(alg);
        Object rateObj = modelConfig.getParams().get(CommonConstants.LEARNING_RATE);
        if(rateObj instanceof Double) {
            rate = (Double) rateObj;
        } else if(rateObj instanceof Integer) {
            // change like this, because user may set it as integer
            rate = ((Integer) rateObj).doubleValue();
        } else if(rateObj instanceof Float) {
            rate = ((Float) rateObj).doubleValue();
        }

        if(toLoggingProcess)
            LOG.info("    - Learning Algorithm: " + learningAlgMap.get(alg));
        if(alg.equals("Q") || alg.equals("B") || alg.equals("M")) {
            if(toLoggingProcess)
                LOG.info("    - Learning Rate: " + rate);
        }

        if(alg.equals("B")) {
            return new Backpropagation(network, trainSet, rate, 0);
        } else if(alg.equals("Q")) {
            return new QuickPropagation(network, trainSet, rate);
        } else if(alg.equals("M")) {
            return new ManhattanPropagation(network, trainSet, rate);
        } else if(alg.equals("R")) {
            return new ResilientPropagation(network, trainSet);
        } else if(alg.equals("S")) {
            return new ScaledConjugateGradient(network, trainSet);
        } else {
            return null;
        }
    }

    private double getValidSetError() {
        // return calculateMSE(this.network, this.validSet);
        return calculateMSEParallel(this.network, this.validSet);
    }

    public double calculateMSEParallel(BasicNetwork network, MLDataSet dataSet) {
        int numRecords = (int) dataSet.getRecordCount();
        assert numRecords > 0;

        // setup workers
        final DetermineWorkload determine = new DetermineWorkload(0, numRecords);
        // nice little workaround
        MSEWorker[] workers = new MSEWorker[determine.getThreadCount()];

        int index = 0;
        TaskGroup group = EngineConcurrency.getInstance().createTaskGroup();
        for(final IntRange r: determine.calculateWorkers()) {
            workers[index++] = new MSEWorker((BasicNetwork) network.clone(), dataSet.openAdditional(), r.getLow(),
                    r.getHigh());
        }

        for(final MSEWorker worker: workers) {
            EngineConcurrency.getInstance().processTask(worker, group);
        }
        group.waitForComplete();

        double totalError = 0;
        for(final MSEWorker worker: workers) {
            totalError += worker.getTotalError();
        }
        return totalError / numRecords;
    }

    private void saveNN() throws IOException {
        if(!toPersistentModel) {
            return;
        }

        File folder = new File(pathFinder.getModelsPath(SourceType.LOCAL));
        if(!folder.exists()) {
            FileUtils.forceMkdir(folder);
        }
        EncogDirectoryPersistence.saveObject(new File(folder, "model" + this.trainerID + ".nn"), network);
    }

    private void saveTmpNN(int epoch) throws IOException {
        if(!toPersistentModel) {
            return;
        }

        File tmpFolder = new File(pathFinder.getTmpModelsPath(SourceType.LOCAL));
        if(!tmpFolder.exists()) {
            FileUtils.forceMkdir(tmpFolder);
        }

        EncogDirectoryPersistence.saveObject(new File(tmpFolder, "model" + trainerID + "-" + epoch + ".nn"), network);
    }

    public MLDataSet getValidSet() {
        return validSet;
    }

    public Double getBaseMSE() {
        if(baseMSE == null) {
            LOG.error("baseMSE is not available. Run train() First!");
            return null;
        }
        return baseMSE;
    }

    private void loadWeightsInput(int numWeights) {
        try {
            File file = new File("./init" + this.trainerID + ".json");
            if(!file.exists()) {

                ModelInitInputObject io = new ModelInitInputObject();
                io.setWeights(randomSetWeights(numWeights));
                io.setNumWeights(numWeights);

                setWeights(io.getWeights());

                JSONUtils.writeValue(file, io);
            } else {
                BufferedReader reader = ShifuFileUtils.getReader("./init" + this.trainerID + ".json", SourceType.LOCAL);
                ModelInitInputObject io = JSONUtils.readValue(reader, ModelInitInputObject.class);
                if(io == null) {
                    io = new ModelInitInputObject();
                }
                if(io.getNumWeights() != numWeights) {

                    io.setNumWeights(numWeights);
                    io.setWeights(randomSetWeights(numWeights));

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
        if(network == null)
            return;
        int i = 0;
        for(int numLayer = 0; numLayer < network.getLayerCount() - 1; numLayer++) {

            int fromCount = network.getLayerTotalNeuronCount(numLayer);
            int toCount = network.getLayerNeuronCount(numLayer + 1);
            for(int fromNeuron = 0; fromNeuron < fromCount; fromNeuron++) {
                for(int toNeuron = 0; toNeuron < toCount; toNeuron++) {
                    network.setWeight(numLayer, fromNeuron, toNeuron, weights.get(i++));
                }
            }
        }
    }

    private List<Double> randomSetWeights(int numWeights) {
        List<Double> weights = new ArrayList<Double>();
        for(int i = 0; i < numWeights; i++) {
            weights.add(this.random.nextDouble() * 2 * Epsilon - Epsilon);
        }
        return weights;
    }
}