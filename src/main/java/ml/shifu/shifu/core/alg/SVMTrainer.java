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

import org.apache.commons.io.FileUtils;
import org.encog.ml.svm.KernelType;
import org.encog.ml.svm.SVM;
import org.encog.ml.svm.SVMType;
import org.encog.ml.svm.training.SVMTrain;
import org.encog.persist.EncogDirectoryPersistence;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Implementation of AbstractTrainer for support vector machine classification
 */
public class SVMTrainer extends AbstractTrainer {

    public static final String SVM_KERNEL = "Kernel";
    public static final String SVM_GAMMA = "Gamma";
    public static final String SVM_CONST = "Const";

    private SVM svm;
    private static Map<String, KernelType> kernel = new HashMap<String, KernelType>();
    private static Map<String, SVMType> type = new HashMap<String, SVMType>();

    protected Logger log = LoggerFactory.getLogger(SVMTrainer.class);

    static {
        kernel.put("Leaner kernel".toLowerCase(), KernelType.Linear);
        kernel.put("linear".toLowerCase(), KernelType.Linear);
        kernel.put("Poly kernel".toLowerCase(), KernelType.Poly);
        kernel.put("poly", KernelType.Poly);
        kernel.put("Sigmoid kernel".toLowerCase(), KernelType.Sigmoid);
        kernel.put("Sigmoid".toLowerCase(), KernelType.Sigmoid);
        kernel.put("RadialBasisFunction".toLowerCase(), KernelType.RadialBasisFunction);
        kernel.put("RBF".toLowerCase(), KernelType.RadialBasisFunction);

        type.put("classification", SVMType.SupportVectorClassification);
        type.put("regresssion", SVMType.EpsilonSupportVectorRegression);

        // kernel.put(KernelType.Precomputed, "")
    }

    /**
     * SVMTrainer Constructor
     * 
     * @param modelConfig
     *            modelConfig
     * @param trainerID
     *            trainerID
     * @param dryRun
     *            dryRun
     */
    public SVMTrainer(ModelConfig modelConfig, int trainerID, Boolean dryRun) {
        super(modelConfig, trainerID, dryRun);
    }

    @Override
    public double train() throws IOException {
        if(this.trainerID == 0) {
            log.info("Trainer #" + (this.trainerID + 1) + " Using SVM algorithm...");
        }
        encogTrain();
        return 0.0d;
    }

    /**
     * Setup SVM
     */
    private void buildSVM() {
        svm = new SVM(this.trainSet.getInputSize(), SVMType.SupportVectorClassification, kernel.get(modelConfig
                .getParams().get("Kernel")));
    }

    /**
     * using Encog's SVM trainer
     */
    private void encogTrain() {
        buildSVM();

        SVMTrain trainer = new SVMTrain(svm, trainSet);
        trainer.setC((Double) modelConfig.getParams().get("Const"));
        trainer.setGamma((Double) modelConfig.getParams().get("Gamma"));

        if(this.trainerID == 0) {
            log.info("Using kenerl function " + svm.getKernelType());
        }

        SVMRunner runner = new SVMRunner(trainer);
        Thread thread = new Thread(runner);
        thread.start();

        long second = 1000;

        while(!runner.isFinish()) {
            try {
                Thread.sleep(second);
                log.info("Trainer #" + this.trainerID + " is running");
            } catch (InterruptedException e) {
                throw new RuntimeException("Within system interrupted");
            }
        }

        log.info("Trainer #" + this.trainerID + " finish training");

        trainer = runner.trainer;

        log.info("Train #" + this.trainerID + " Error: " + df.format(trainer.getError()) + " Validation Error:"
                + df.format(getValidSetError()));

        saveModel();
    }

    private void saveModel() {
        File folder = new File("./models");
        if(!folder.exists()) {
            try {
                FileUtils.forceMkdir(folder);
            } catch (IOException e) {
                log.error("Failed to create directory: {}", folder.getAbsolutePath());
                e.printStackTrace();
            }
        }
        EncogDirectoryPersistence.saveObject(new File("./models/model" + this.trainerID + ".svm"), svm);
    }

    public double getValidSetError() {
        return svm.calculateError(this.validSet);
    }

    public SVM getSVM() {
        return svm;
    }

    /**
     * SVMtrainer worker
     */
    private static class SVMRunner implements Runnable {

        private SVMTrain trainer;
        private boolean isFinish;

        public SVMRunner(SVMTrain trainer) {
            this.trainer = trainer;
            this.isFinish = false;
        }

        @Override
        public void run() {

            trainer.setFold(1);

            trainer.iteration();

            isFinish = true;
        }

        public boolean isFinish() {
            return isFinish;
        }
    }

}
