package ml.shifu.shifu.core.dvarsel.util;
/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.alg.NNTrainer;
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
import org.encog.util.concurrency.DetermineWorkload;
import org.encog.util.concurrency.EngineConcurrency;
import org.encog.util.concurrency.TaskGroup;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.text.DecimalFormat;
import java.util.Arrays;
import java.util.List;
import java.util.Set;

/**
 * Created on 11/24/2014.
 */
public class NNValidator {
    private static final Logger LOG = LoggerFactory.getLogger(NNValidator.class);
    private static final DecimalFormat df = new DecimalFormat("0.000000");

    private ModelConfig modelConfig;
    private List<ColumnConfig> columnConfigList;
    private Set<Integer> workingColumnIdSet;
    private MLDataSet trainingDataSet;
    private MLDataSet testDataSet;

    private BasicNetwork network;

    private double totalValidationError = 0.0;

    public NNValidator(ModelConfig modelConfig,
                       List<ColumnConfig> columnConfigList,
                       Set<Integer> workingColumnIdSet,
                       MLDataSet trainingDataSet,
                       MLDataSet testDataSet) {
        this.modelConfig = modelConfig;
        this.columnConfigList = columnConfigList;
        this.workingColumnIdSet = workingColumnIdSet;
        this.trainingDataSet = trainingDataSet;
        this.testDataSet = testDataSet;

        buildNetwork();
    }

    public double validate() {
        LOG.info("Using neural network algorithm...");
        LOG.info("\t - Input Size: " + trainingDataSet.getInputSize());
        LOG.info("\t  - Ideal Size: " + trainingDataSet.getIdealSize());

        Propagation mlTrain = getMLTrain();
        if ( mlTrain == null ) {
            throw new RuntimeException("Fail to create propagation!");
        }

        mlTrain.setThreadCount(0);

        int epochs = this.modelConfig.getNumTrainEpochs();
        double minValidationError = Double.MAX_VALUE;
        for (int i = 0; i < epochs; i++) {
            mlTrain.iteration();
            double validationError = (this.testDataSet.getRecordCount() > 0)
                    ? this.calculateValidationError(network) : mlTrain.getError();
            if ( validationError < minValidationError) {
                minValidationError = validationError;
            }

            LOG.info("Epoch #" + (i + 1)
                    + " Train Error: "
                    + df.format(mlTrain.getError())
                    + " Validation Error: "
                    + ((this.testDataSet.getRecordCount() > 0) ? df.format(validationError) : "N/A"));
        }

        mlTrain.finishTraining();
        LOG.info("Trainer for ColumnIdSet - {} is Finished!", this.workingColumnIdSet.toString());

        return minValidationError;
    }

    private void buildNetwork() {
        network = new BasicNetwork();
        network.addLayer(new BasicLayer(new ActivationLinear(), true, trainingDataSet.getInputSize()));

        int numLayers = (Integer) this.modelConfig.getParams()
                .get(NNTrainer.NUM_HIDDEN_LAYERS);
        List<String> actFunc = (List<String>) this.modelConfig.getParams()
                .get(NNTrainer.ACTIVATION_FUNC);
        List<Integer> hiddenNodeList = (List<Integer>) this.modelConfig.getParams()
                .get(NNTrainer.NUM_HIDDEN_NODES);

        if ( numLayers != 0 && (numLayers != actFunc.size() || numLayers != hiddenNodeList.size())) {
            throw new RuntimeException("the number of layer do not equal to the number of activation function or the function list and node list empty");
        }

        LOG.info("\t - total " + numLayers + " layers, each layers are: " + Arrays.toString(hiddenNodeList.toArray()) + " the activation function are: " + Arrays.toString(actFunc.toArray()));

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
                LOG.info("Unsupported activation function: " + func + " !! Set this layer activation function to be Sigmoid ");
                network.addLayer(new BasicLayer(new ActivationSigmoid(), true, numHiddenNode));
            }
        }

        network.addLayer(new BasicLayer(new ActivationSigmoid(), false, trainingDataSet.getIdealSize()));
        network.getStructure().finalizeStructure();
        network.reset();
    }

    private Propagation getMLTrain() {
        String alg = (String) this.modelConfig.getParams().get(NNTrainer.PROPAGATION);
        if (!(NNTrainer.defaultLearningRate.containsKey(alg))) {
            throw new RuntimeException("Learning Algorithm is not valid: " + alg);
        }

        double rate = NNTrainer.defaultLearningRate.get(alg);
        Object rateObj = this.modelConfig.getParams().get(NNTrainer.LEARNING_RATE);
        if (rateObj instanceof Double) {
            rate = (Double) rateObj;
        } else if (rateObj instanceof Integer) {
            // change like this, because user may set it as integer
            rate = ((Integer) rateObj).doubleValue();
        } else if (rateObj instanceof Float) {
            rate = ((Float) rateObj).doubleValue();
        }

        LOG.info("\t - Learning Algorithm: " + NNTrainer.learningAlgMap.get(alg));
        if (alg.equals("Q") || alg.equals("B") || alg.equals("M")) {
            LOG.info("\t - Learning Rate: " + rate);
        }

        if (alg.equals("B")) {
            return new Backpropagation(network, trainingDataSet, rate, 0);
        } else if (alg.equals("Q")) {
            return new QuickPropagation(network, trainingDataSet, rate);
        } else if (alg.equals("M")) {
            return new ManhattanPropagation(network, trainingDataSet, rate);
        } else if (alg.equals("R")) {
            return new ResilientPropagation(network, trainingDataSet);
        } else if (alg.equals("S")) {
            return new ScaledConjugateGradient(network, trainingDataSet);
        } else {
            return null;
        }
    }

    public double calculateValidationError(BasicNetwork network) {
        totalValidationError = 0.0;

        int numRecords = (int) testDataSet.getRecordCount();
        assert numRecords > 0;

        // setup workers
        final DetermineWorkload determine = new DetermineWorkload(0, numRecords);

        // nice little workaround
        MSEWorker[] workers = new MSEWorker[determine.getThreadCount()];

        int index = 0;
        TaskGroup group = EngineConcurrency.getInstance().createTaskGroup();
        for (final IntRange r : determine.calculateWorkers()) {
            workers[index++] = new MSEWorker((BasicNetwork)network.clone(), this,
                    testDataSet.openAdditional(), r.getLow(), r.getHigh());
        }

        for (final MSEWorker worker : workers) {
            EngineConcurrency.getInstance().processTask(worker, group);
        }

        group.waitForComplete();
        return this.totalValidationError / numRecords;
    }

    public final void report(double error) {
        synchronized (this) {
            this.totalValidationError += error;
        }
    }
}
