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
package ml.shifu.shifu.core.dtrain;

import java.io.IOException;

import ml.shifu.shifu.core.dtrain.nn.NNParams;

import org.encog.engine.network.activation.ActivationFunction;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.error.LinearErrorFunction;
import org.encog.neural.flat.FlatNetwork;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.back.Backpropagation;
import org.encog.neural.networks.training.propagation.manhattan.ManhattanPropagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.benchmark.RandomTrainingFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.BeforeTest;
import org.testng.annotations.Test;

public class DTrainTest {

    public static final int INPUT_COUNT = 1000;
    public static final int HIDDEN_COUNT = 20;
    public static final int OUTPUT_COUNT = 1;
    public BasicNetwork network;
    public MLDataSet training;

    public static final int NUM_EPOCHS = 20;

    public double rate = 0.5;

    public int numSplit = 24;

    public double[] weights;

    private final static Logger log = LoggerFactory.getLogger(DTrainTest.class);

    @BeforeTest
    public void setup() {
        network = new BasicNetwork();
        network.addLayer(new BasicLayer(DTrainTest.INPUT_COUNT));
        network.addLayer(new BasicLayer(DTrainTest.HIDDEN_COUNT));
        network.addLayer(new BasicLayer(DTrainTest.OUTPUT_COUNT));
        network.getStructure().finalizeStructure();
        network.reset();

        weights = network.getFlat().getWeights();

        training = RandomTrainingFactory.generate(1000, 10000, INPUT_COUNT, OUTPUT_COUNT, -1, 1);
    }

    public Gradient initGradient(MLDataSet training) {

        FlatNetwork flat = network.getFlat().clone();

        // copy Propagation from encog
        double[] flatSpot = new double[flat.getActivationFunctions().length];
        for(int i = 0; i < flat.getActivationFunctions().length; i++) {
            final ActivationFunction af = flat.getActivationFunctions()[i];

            if(af instanceof ActivationSigmoid) {
                flatSpot[i] = 0.1;
            } else {
                flatSpot[i] = 0.0;
            }
        }

        return new Gradient(flat, training.openAdditional(), training, flatSpot, new LinearErrorFunction(), false);
    }

    @Test
    public void quickTest() throws IOException {

        double[] gradientError = new double[NUM_EPOCHS];
        double[] ecogError = new double[NUM_EPOCHS];

        network.reset();
        weights = network.getFlat().getWeights();

        MLDataSet[] subsets = splitDataSet(training);
        Gradient[] workers = new Gradient[numSplit];

        Weight weightCalculator = null;

        for(int i = 0; i < workers.length; i++) {
            workers[i] = initGradient(subsets[i]);
            workers[i].setWeights(weights);
        }

        log.info("Running QuickPropagtaion testing! ");
        NNParams globalParams = new NNParams();
        globalParams.setWeights(weights);

        for(int i = 0; i < NUM_EPOCHS; i++) {

            double error = 0.0;

            // each worker do the job
            for(int j = 0; j < workers.length; j++) {
                workers[j].run();
                error += workers[j].getError();
            }

            gradientError[i] = error / workers.length;

            log.info("The #" + i + " training error: " + gradientError[i]);

            // master
            globalParams.reset();

            for(int j = 0; j < workers.length; j++) {
                globalParams.accumulateGradients(workers[j].getGradients());
                globalParams.accumulateTrainSize(subsets[j].getRecordCount());
            }

            if(weightCalculator == null) {
                weightCalculator = new Weight(globalParams.getGradients().length, globalParams.getTrainSize(),
                        this.rate, DTrainUtils.QUICK_PROPAGATION, 0, RegulationLevel.NONE, 0d);
            }

            double[] interWeight = weightCalculator.calculateWeights(globalParams.getWeights(),
                    globalParams.getGradients(), -1);

            globalParams.setWeights(interWeight);

            // set weights
            for(int j = 0; j < workers.length; j++) {
                workers[j].setWeights(interWeight);
            }
        }

        // encog
        network.reset();
        // NNUtils.randomize(numSplit, weights);
        network.getFlat().setWeights(weights);

        Propagation p = null;
        p = new QuickPropagation(network, training, rate);
        // p = new ManhattanPropagation(network, training, rate);
        p.setThreadCount(numSplit);

        for(int i = 0; i < NUM_EPOCHS; i++) {
            p.iteration(1);
            // System.out.println("the #" + i + " training error: " + p.getError());
            ecogError[i] = p.getError();
        }

        // assert
        double diff = 0.0;
        for(int i = 0; i < NUM_EPOCHS; i++) {
            diff += Math.abs(ecogError[i] - gradientError[i]);
        }

        Assert.assertTrue(diff / NUM_EPOCHS < 0.1);

    }

    private MLDataSet[] splitDataSet(MLDataSet data) {

        MLDataSet[] subsets = new MLDataSet[numSplit];

        for(int i = 0; i < subsets.length; i++) {
            subsets[i] = new BasicMLDataSet();
        }

        for(int i = 0; i < data.getRecordCount(); i++) {
            MLDataPair pair = BasicMLDataPair.createPair(INPUT_COUNT, OUTPUT_COUNT);
            data.getRecord(i, pair);
            subsets[i % numSplit].add(pair);
        }

        return subsets;
    }

    @Test
    public void manhantTest() throws IOException {
        double[] gradientError = new double[NUM_EPOCHS];
        double[] ecogError = new double[NUM_EPOCHS];

        network.reset();
        weights = network.getFlat().getWeights();

        MLDataSet[] subsets = splitDataSet(training);
        Gradient[] workers = new Gradient[numSplit];

        Weight weightCalculator = null;

        for(int i = 0; i < workers.length; i++) {
            workers[i] = initGradient(subsets[i]);
            workers[i].setWeights(weights);
        }

        NNParams globalParams = new NNParams();
        globalParams.setWeights(weights);

        log.info("Starting manhattan propagation testing!");

        for(int i = 0; i < NUM_EPOCHS; i++) {

            double error = 0.0;

            // each worker do the job
            for(int j = 0; j < workers.length; j++) {
                workers[j].run();
                error += workers[j].getError();
            }

            gradientError[i] = error / workers.length;

            log.info("The #" + i + " training error: " + gradientError[i]);

            // master
            globalParams.reset();

            for(int j = 0; j < workers.length; j++) {
                globalParams.accumulateGradients(workers[j].getGradients());
                globalParams.accumulateTrainSize(subsets[j].getRecordCount());
            }

            if(weightCalculator == null) {
                weightCalculator = new Weight(globalParams.getGradients().length, globalParams.getTrainSize(),
                        this.rate, DTrainUtils.MANHATTAN_PROPAGATION, 0, RegulationLevel.NONE, 0d);
            }

            double[] interWeight = weightCalculator.calculateWeights(globalParams.getWeights(),
                    globalParams.getGradients(), -1);

            globalParams.setWeights(interWeight);

            // set weights
            for(int j = 0; j < workers.length; j++) {
                workers[j].setWeights(interWeight);
            }
        }

        // encog
        network.reset();
        // NNUtils.randomize(numSplit, weights);
        network.getFlat().setWeights(weights);

        Propagation p = null;
        p = new ManhattanPropagation(network, training, rate);
        p.setThreadCount(numSplit);

        for(int i = 0; i < NUM_EPOCHS; i++) {
            p.iteration(1);
            // System.out.println("the #" + i + " training error: " + p.getError());
            ecogError[i] = p.getError();
        }

        // assert
        double diff = 0.0;
        for(int i = 0; i < NUM_EPOCHS; i++) {
            diff += Math.abs(ecogError[i] - gradientError[i]);
        }

        Assert.assertTrue(diff / NUM_EPOCHS < 0.3);
    }

    @Test
    public void backTest() {
        double[] gradientError = new double[NUM_EPOCHS];
        double[] ecogError = new double[NUM_EPOCHS];

        network.reset();
        weights = network.getFlat().getWeights();

        MLDataSet[] subsets = splitDataSet(training);
        Gradient[] workers = new Gradient[numSplit];

        Weight weightCalculator = null;

        for(int i = 0; i < workers.length; i++) {
            workers[i] = initGradient(subsets[i]);
            workers[i].setWeights(weights);
        }

        log.info("Starting back propagation testing!");
        NNParams globalParams = new NNParams();
        globalParams.setWeights(weights);

        for(int i = 0; i < NUM_EPOCHS; i++) {

            double error = 0.0;

            // each worker do the job
            for(int j = 0; j < workers.length; j++) {
                workers[j].run();
                error += workers[j].getError();
            }

            gradientError[i] = error / workers.length;

            log.info("The #" + i + " training error: " + gradientError[i]);

            // master
            globalParams.reset();

            for(int j = 0; j < workers.length; j++) {
                globalParams.accumulateGradients(workers[j].getGradients());
                globalParams.accumulateTrainSize(subsets[j].getRecordCount());
            }

            if(weightCalculator == null) {
                weightCalculator = new Weight(globalParams.getGradients().length, globalParams.getTrainSize(),
                        this.rate, DTrainUtils.BACK_PROPAGATION, 0, RegulationLevel.NONE, 0d);
            }

            double[] interWeight = weightCalculator.calculateWeights(globalParams.getWeights(),
                    globalParams.getGradients(), -1);

            globalParams.setWeights(interWeight);

            // set weights
            for(int j = 0; j < workers.length; j++) {
                workers[j].setWeights(interWeight);
            }
        }

        // encog
        network.reset();
        // NNUtils.randomize(numSplit, weights);
        network.getFlat().setWeights(weights);

        Propagation p = null;
        p = new Backpropagation(network, training, rate, 0.5);
        p.setThreadCount(numSplit);

        for(int i = 0; i < NUM_EPOCHS; i++) {
            p.iteration(1);
            // System.out.println("the #" + i + " training error: " + p.getError());
            ecogError[i] = p.getError();
        }

        // assert
        double diff = 0.0;
        for(int i = 0; i < NUM_EPOCHS; i++) {
            diff += Math.abs(ecogError[i] - gradientError[i]);
        }

        Assert.assertTrue(diff / NUM_EPOCHS < 0.2);
    }

    @Test
    public void resilientPropagationTest() {
        double[] gradientError = new double[NUM_EPOCHS];
        double[] ecogError = new double[NUM_EPOCHS];

        network.reset();
        weights = network.getFlat().getWeights();

        MLDataSet[] subsets = splitDataSet(training);
        Gradient[] workers = new Gradient[numSplit];

        Weight weightCalculator = null;

        for(int i = 0; i < workers.length; i++) {
            workers[i] = initGradient(subsets[i]);
            workers[i].setWeights(weights);
        }

        log.info("Starting resilient propagation testing!");
        NNParams globalParams = new NNParams();
        globalParams.setWeights(weights);

        for(int i = 0; i < NUM_EPOCHS; i++) {

            double error = 0.0;

            // each worker do the job
            for(int j = 0; j < workers.length; j++) {
                workers[j].run();
                error += workers[j].getError();
            }

            gradientError[i] = error / workers.length;

            log.info("The #" + i + " training error: " + gradientError[i]);

            // master
            globalParams.reset();

            for(int j = 0; j < workers.length; j++) {
                globalParams.accumulateGradients(workers[j].getGradients());
                globalParams.accumulateTrainSize(subsets[j].getRecordCount());
            }

            if(weightCalculator == null) {
                weightCalculator = new Weight(globalParams.getGradients().length, globalParams.getTrainSize(),
                        this.rate, DTrainUtils.RESILIENTPROPAGATION, 0, RegulationLevel.NONE, 0d);
            }

            double[] interWeight = weightCalculator.calculateWeights(globalParams.getWeights(),
                    globalParams.getGradients(), -1);

            globalParams.setWeights(interWeight);

            // set weights
            for(int j = 0; j < workers.length; j++) {
                workers[j].setWeights(interWeight);
            }
        }

        // encog
        network.reset();
        // NNUtils.randomize(numSplit, weights);
        network.getFlat().setWeights(weights);

        Propagation p = null;
        p = new ResilientPropagation(network, training);
        p.setThreadCount(numSplit);

        for(int i = 0; i < NUM_EPOCHS; i++) {
            p.iteration(1);
            ecogError[i] = p.getError();
        }

        // assert
        double diff = 0.0;
        for(int i = 0; i < NUM_EPOCHS; i++) {
            diff += Math.abs(ecogError[i] - gradientError[i]);
        }

        Assert.assertTrue(diff / NUM_EPOCHS < 0.2);
    }
}
