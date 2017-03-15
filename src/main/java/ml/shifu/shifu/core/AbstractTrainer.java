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
package ml.shifu.shifu.core;

import ml.shifu.shifu.container.ModelInitInputObject;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.JSONUtils;
import org.encog.ml.BasicML;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.ml.data.buffer.BufferedMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;


/**
 * Abstract Trainer
 */
public abstract class AbstractTrainer {
    /**
     * Log for abstract trainer
     */
    protected static final Logger log = LoggerFactory.getLogger(AbstractTrainer.class);

    /**
     * formatter
     */
    protected static final DecimalFormat df = new DecimalFormat("0.000000");

    /**
     * randomizer
     */
    protected Random random;

    /**
     * model config instance
     */
    protected ModelConfig modelConfig;

    /**
     * trainer id, identify the trainer
     */
    protected int trainerID = 0;

    /**
     * dry run flag
     */
    protected Boolean dryRun = false;

    /**
     * cross validation rate
     */
    protected Double crossValidationRate;

    /**
     * sample rate
     */
    protected Double baggingSampleRate;

    /**
     * training option, M: memory, D: disk
     */
    protected String trainingOption = "M";

    /**
     * training set instance
     */
    protected MLDataSet trainSet;

    /**
     * validation set instance
     */
    protected MLDataSet validSet;

    /**
     * base standard error value
     */
    protected Double baseMSE = null;

    /**
     * path finder to locate file
     */
    protected PathFinder pathFinder = null;

    public AbstractTrainer(ModelConfig modelConfig, int trainerID, Boolean dryRun) {
        this.random = new Random(System.currentTimeMillis() + trainerID);

        this.modelConfig = modelConfig;
        this.trainerID = trainerID;
        this.dryRun = dryRun;

        crossValidationRate = this.modelConfig.getValidSetRate();
        if (crossValidationRate == null) {
            crossValidationRate = 0.2;
        }

        baggingSampleRate = this.modelConfig.getBaggingSampleRate();
        if (baggingSampleRate == null) {
            baggingSampleRate = 0.8;
        }

        pathFinder = new PathFinder(modelConfig);
    }

    /*
     * Set up the training dataset and validation dataset
     */
    public void setDataSet(MLDataSet masterDataSet) throws IOException {
        log.info("Setting Data Set...");

        MLDataSet sampledDataSet;

        if (this.trainingOption.equalsIgnoreCase("M")) {
            log.info("Loading to Memory ...");
            sampledDataSet = new BasicMLDataSet();
            this.trainSet = new BasicMLDataSet();
            this.validSet = new BasicMLDataSet();
        } else if (this.trainingOption.equalsIgnoreCase("D")) {
            log.info("Loading to Disk ...");
            sampledDataSet = new BufferedMLDataSet(new File(Constants.TMP, "sampled.egb"));
            this.trainSet = new BufferedMLDataSet(new File(Constants.TMP, "train.egb"));
            this.validSet = new BufferedMLDataSet(new File(Constants.TMP, "valid.egb"));

            int inputSize = masterDataSet.getInputSize();
            int idealSize = masterDataSet.getIdealSize();
            ((BufferedMLDataSet) sampledDataSet).beginLoad(inputSize, idealSize);
            ((BufferedMLDataSet) trainSet).beginLoad(inputSize, idealSize);
            ((BufferedMLDataSet) validSet).beginLoad(inputSize, idealSize);
        } else {
            throw new RuntimeException("Training Option is not Valid: " + this.trainingOption);
        }

        // Encog 3.1
        // int masterSize = masterDataSet.size();

        // Encog 3.0
        int masterSize = (int) masterDataSet.getRecordCount();

        if (!modelConfig.isFixInitialInput()) {
            // Bagging
            if (modelConfig.isBaggingWithReplacement()) {
                // Bagging With Replacement
                int sampledSize = (int) (masterSize * baggingSampleRate);
                for (int i = 0; i < sampledSize; i++) {
                    // Encog 3.1
                    // sampledDataSet.add(masterDataSet.get(random.nextInt(masterSize)));

                    // Encog 3.0
                    double[] input = new double[masterDataSet.getInputSize()];
                    double[] ideal = new double[1];
                    MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));

                    masterDataSet.getRecord(random.nextInt(masterSize), pair);
                    sampledDataSet.add(pair);
                }
            } else {
                // Bagging Without Replacement
                for (MLDataPair pair : masterDataSet) {
                    if (random.nextDouble() < baggingSampleRate) {
                        sampledDataSet.add(pair);
                    }
                }
            }
        } else {
            List<Integer> list = loadSampleInput((int) (masterSize * baggingSampleRate), masterSize,
                    modelConfig.isBaggingWithReplacement());
            for (Integer i : list) {
                double[] input = new double[masterDataSet.getInputSize()];
                double[] ideal = new double[1];
                MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));
                masterDataSet.getRecord(i, pair);
                sampledDataSet.add(pair);
            }
        }

        if (this.trainingOption.equalsIgnoreCase("D")) {
            ((BufferedMLDataSet) sampledDataSet).endLoad();
        }
        // Cross Validation
        log.info("Generating Training Set and Validation Set ...");

        if (!modelConfig.isFixInitialInput()) {
            // Encog 3.0
            for (int i = 0; i < sampledDataSet.getRecordCount(); i++) {

                double[] input = new double[sampledDataSet.getInputSize()];
                double[] ideal = new double[1];
                MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));

                sampledDataSet.getRecord(i, pair);

                if (random.nextDouble() > crossValidationRate) {
                    trainSet.add(pair);
                } else {
                    validSet.add(pair);
                }
            }
        } else {
            long sampleSize = sampledDataSet.getRecordCount();
            long trainSetSize = (long) (sampleSize * (1 - crossValidationRate));
            int i = 0;
            for (; i < trainSetSize; i++) {
                double[] input = new double[sampledDataSet.getInputSize()];
                double[] ideal = new double[1];
                MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));

                sampledDataSet.getRecord(i, pair);
                trainSet.add(pair);
            }

            for (; i < sampleSize; i++) {
                double[] input = new double[sampledDataSet.getInputSize()];
                double[] ideal = new double[1];
                MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));

                sampledDataSet.getRecord(i, pair);
                validSet.add(pair);
            }
        }

        if (this.trainingOption.equalsIgnoreCase("D")) {
            ((BufferedMLDataSet) trainSet).endLoad();
            ((BufferedMLDataSet) validSet).endLoad();
        }

        log.info("    - # Records of the Master Data Set: " + masterSize);
        log.info("    - Bagging Sample Rate: " + baggingSampleRate);
        log.info("    - Bagging With Replacement: " + modelConfig.isBaggingWithReplacement());
        log.info("    - # Records of the Selected Data Set: " + sampledDataSet.getRecordCount());
        log.info("        - Cross Validation Rate: " + crossValidationRate);
        log.info("        - # Records of the Training Set: " + this.getTrainSetSize());
        log.info("        - # Records of the Validation Set: " + this.getValidSetSize());

    }

    /**
     * get the training data set size
     *
     * @return number of size
     */
    public int getTrainSetSize() {
        // Encog 3.1
        // return trainSet.size();

        // Encog 3.0
        return (int) trainSet.getRecordCount();
    }

    /**
     * get the training dataset
     *
     * @return the trainSet
     */
    public MLDataSet getTrainSet() {
        return trainSet;
    }

    /**
     * @param trainSet the trainSet to set
     */
    public void setTrainSet(MLDataSet trainSet) {
        this.trainSet = trainSet;
    }

    /**
     * @param validSet the validSet to set
     */
    public void setValidSet(MLDataSet validSet) {
        this.validSet = validSet;
    }

    /**
     * get the validation set size
     *
     * @return the validation set number
     */
    public int getValidSetSize() {
        // Encog 3.1
        // return validSet.size();

        // Encog 3.0
        return (int) validSet.getRecordCount();
    }

    /*
     * set the training option, M/D
     */
    public void setTrainingOption(String trainingOption) {
        this.trainingOption = trainingOption;
    }

    /**
     * get the validation dataset
     *
     * @return the validation dataset
     */
    public MLDataSet getValidSet() {
        return validSet;
    }

    /*
     * load/save the sampling data from pre-initialization file
     */
    private List<Integer> loadSampleInput(int sampleSize, int masterSize, boolean replaceable) throws IOException {
        List<Integer> list = null;

        File file = new File("./init" + trainerID + ".json");
        if (!file.exists()) {

            list = randomSetSampleIndex(sampleSize, masterSize, replaceable);

            ModelInitInputObject io = new ModelInitInputObject();
            io.setNumSample(sampleSize);
            io.setSampleIndex(list);

            JSONUtils.writeValue(file, io);

        } else {
            BufferedReader reader = ShifuFileUtils.getReader("./init" + trainerID + ".json", SourceType.LOCAL);
            ModelInitInputObject io = JSONUtils.readValue(reader, ModelInitInputObject.class);

            if (io == null) {
                io = new ModelInitInputObject();
            }
            if (io.getNumSample() != sampleSize) {

                list = randomSetSampleIndex(sampleSize, masterSize, replaceable);

                io.setNumSample(sampleSize);
                io.setSampleIndex(list);

                JSONUtils.writeValue(file, io);
            } else {
                list = io.getSampleIndex();
            }
            reader.close();
        }

        return list;
    }

    /*
     * randomizer the input data
     */
    private List<Integer> randomSetSampleIndex(int sampleSize, int masterSize, boolean replaceable) {
        List<Integer> list = new ArrayList<Integer>();
        if (replaceable) {
            for (int i = 0; i < sampleSize; i++) {
                list.add(random.nextInt(masterSize));
            }
        } else {
            for (int i = 0; i < masterSize; i++) {
                if (random.nextDouble() < baggingSampleRate) {
                    list.add(i);
                }
            }
        }
        return list;
    }

    /*
     * reset the weights in trainer
     */
    public void resetParams(BasicML classifier) {
        if (modelConfig.isFixInitialInput()) {

        } else {
            if (modelConfig.getAlgorithm() == "NN" || modelConfig.getAlgorithm() == "LR") {
                ((BasicNetwork) classifier).reset();
            }
        }
    }

    /*
     * A training start function, and print the training error and validation errors
     */
    public abstract double train() throws IOException;

    /*
     * non-synchronously version update error
     *
     * @return the standard error
     */
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

    public Double getBaseMSE() {
        return baseMSE;
    }

    public void setBaseMSE(Double baseMSE) {
        this.baseMSE = baseMSE;
    }

}
