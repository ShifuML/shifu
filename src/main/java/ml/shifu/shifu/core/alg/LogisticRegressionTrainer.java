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

import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.AbstractTrainer;
import ml.shifu.shifu.core.ConvergeJudger;

import org.apache.commons.io.FileUtils;
import org.encog.engine.network.activation.ActivationLinear;
import org.encog.engine.network.activation.ActivationSigmoid;
import org.encog.ml.data.MLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.persist.EncogDirectoryPersistence;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;

/**
 * Implementation of AbstractTrainer for LogisticRegression
 */
public class LogisticRegressionTrainer extends AbstractTrainer {

    private static final Logger LOG = LoggerFactory.getLogger(LogisticRegressionTrainer.class);

    public static final String LEARNING_RATE = "LearningRate";

    private BasicNetwork classifier;

    /**
     * Convergence judger instance for convergence criteria checking.
     */
    private ConvergeJudger judger = new ConvergeJudger();

    public LogisticRegressionTrainer(ModelConfig modelConfig, int trainerID, Boolean dryRun) {
        super(modelConfig, trainerID, dryRun);

    }

    public void setDataSet(MLDataSet masterDataSet) throws IOException {
        super.setDataSet(masterDataSet);
    }

    /**
     * {@inheritDoc}
     * <p>
     * no <code>regularization</code>
     * <p>
     * Regular will be provide later
     * <p>
     * 
     * @throws IOException
     *             e
     */
    @Override
    public double train() throws IOException {
        classifier = new BasicNetwork();

        classifier.addLayer(new BasicLayer(new ActivationLinear(), true, trainSet.getInputSize()));
        classifier.addLayer(new BasicLayer(new ActivationSigmoid(), false, trainSet.getIdealSize()));
        classifier.getStructure().finalizeStructure();

        // resetParams(classifier);
        classifier.reset();

        // Propagation mlTrain = getMLTrain();
        Propagation propagation = new QuickPropagation(classifier, trainSet, (Double) modelConfig.getParams().get(
                "LearningRate"));
        int epochs = modelConfig.getNumTrainEpochs();

        // Get convergence threshold from modelConfig.
        double threshold = modelConfig.getTrain().getConvergenceThreshold() == null ? 0.0 : modelConfig.getTrain()
                .getConvergenceThreshold().doubleValue();
        String formatedThreshold = df.format(threshold);

        LOG.info("Using " + (Double) modelConfig.getParams().get("LearningRate") + " training rate");

        for(int i = 0; i < epochs; i++) {
            propagation.iteration();
            double trainError = propagation.getError();
            double validError = classifier.calculateError(this.validSet);

            LOG.info("Epoch #" + (i + 1) + " Train Error:" + df.format(trainError) + " Validation Error:"
                    + df.format(validError));

            // Convergence judging.
            double avgErr = (trainError + validError) / 2;

            if(judger.judge(avgErr, threshold)) {
                LOG.info("Trainer-{}> Epoch #{} converged! Average Error: {}, Threshold: {}", trainerID, (i + 1),
                        df.format(avgErr), formatedThreshold);
                break;
            }
        }
        propagation.finishTraining();

        LOG.info("#" + this.trainerID + " finish training");

        saveLR();

        return 0.0d;
    }

    private void saveLR() throws IOException {
        File folder = new File("./models");
        if(!folder.exists()) {
            FileUtils.forceMkdir(folder);
        }
        EncogDirectoryPersistence.saveObject(new File("./models/model" + this.trainerID + ".lr"), classifier);
    }

    public BasicNetwork getClassifier() {
        return classifier;
    }
}
