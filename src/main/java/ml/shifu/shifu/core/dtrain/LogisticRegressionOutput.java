/*
 * Copyright [2013-2014] eBay Software Foundation
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
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

import ml.shifu.guagua.master.BasicMasterInterceptor;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link LogisticRegressionOutput} is used to write the final model output to file system.
 */
public class LogisticRegressionOutput extends
        BasicMasterInterceptor<LogisticRegressionParams, LogisticRegressionParams> {

    private static final Logger LOG = LoggerFactory.getLogger(LogisticRegressionOutput.class);

    private static final double EPSILON = 0.0000001;

    private String trainerId;

    private String tmpModelsFolder;

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    /**
     * Whether the training is dry training.
     */
    private boolean isDry;

    /**
     * A flag: whether params initialized.
     */
    private AtomicBoolean isInit = new AtomicBoolean(false);

    /**
     * The minimum test error during model training
     */
    private double minTestError = Double.MAX_VALUE;

    /**
     * The best weights that we meet
     */
    private double[] optimizeddWeights = null;

    /**
     * Progress output stream which is used to write progress to that HDFS file. Should be closed in
     * {@link #postApplication(MasterContext)}.
     */
    private FSDataOutputStream progressOutput = null;

    @Override
    public void preApplication(MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        init(context);
    }

    @Override
    public void postIteration(final MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        if(this.isDry) {
            // for dry mode, we don't save models files.
            return;
        }

        double currentError = ((modelConfig.getTrain().getValidSetRate() < EPSILON) ? context.getMasterResult()
                .getTrainError() : context.getMasterResult().getTestError());

        // save the weights according the error decreasing
        if(currentError < this.minTestError) {
            this.minTestError = currentError;
            this.optimizeddWeights = context.getMasterResult().getParameters();
        }

        // save tmp to hdfs according to raw trainer logic
        if(context.getCurrentIteration() % DTrainUtils.tmpModelFactor(context.getTotalIteration()) == 0) {
            Thread tmpNNThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    saveTmpModelToHDFS(context.getCurrentIteration(), context.getMasterResult().getParameters());
                }
            }, "saveTmpModelToHDFS thread");
            tmpNNThread.setDaemon(true);
            tmpNNThread.start();
        }

        updateProgressLog(context);
    }

    @SuppressWarnings("deprecation")
    private void updateProgressLog(final MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        int currentIteration = context.getCurrentIteration();
        if(currentIteration == 1) {
            // first iteration is used for training preparation
            return;
        }
        String progress = new StringBuilder(200).append("    Trainer ").append(this.trainerId).append(" Epoch #")
                .append(currentIteration - 1).append(" Train Error:").append(context.getMasterResult().getTrainError())
                .append(" Validation Error:").append(context.getMasterResult().getTestError()).append("\n").toString();
        try {
            LOG.debug("Writing progress results to {} {}", context.getCurrentIteration(), progress.toString());
            this.progressOutput.write(progress.getBytes("UTF-8"));
            this.progressOutput.flush();
            this.progressOutput.sync();
        } catch (IOException e) {
            LOG.error("Error in write progress log:", e);
        }
    }

    @Override
    public void postApplication(MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        IOUtils.closeStream(this.progressOutput);

        // for dry mode, we don't save models files.
        if(this.isDry) {
            return;
        }

        if(optimizeddWeights == null) {
            optimizeddWeights = context.getMasterResult().getParameters();
        }

        Path out = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
        writeModelWeightsToFileSystem(optimizeddWeights, out);
    }

    /**
     * Save tmp nn model to HDFS.
     */
    private void saveTmpModelToHDFS(int iteration, double[] weights) {
        Path out = new Path(DTrainUtils.getTmpModelName(this.tmpModelsFolder, this.trainerId, iteration, modelConfig
                .getTrain().getAlgorithm().toLowerCase()));
        writeModelWeightsToFileSystem(weights, out);
    }

    private void init(MasterContext<LogisticRegressionParams, LogisticRegressionParams> context) {
        this.isDry = Boolean.TRUE.toString().equals(context.getProps().getProperty(NNConstants.NN_DRY_TRAIN));

        if(this.isDry) {
            return;
        }
        if(isInit.compareAndSet(false, true)) {
            loadConfigFiles(context.getProps());
            this.trainerId = context.getProps().getProperty(NNConstants.NN_TRAINER_ID);
            this.tmpModelsFolder = context.getProps().getProperty(NNConstants.NN_TMP_MODELS_FOLDER);
        }

        try {
            Path progressLog = new Path(context.getProps().getProperty(NNConstants.NN_PROGRESS_FILE));
            this.progressOutput = FileSystem.get(new Configuration()).create(progressLog);
        } catch (IOException e) {
            LOG.error("Error in create progress log:", e);
        }
    }

    /**
     * Load all configurations for modelConfig and columnConfigList from source type. Use null check to make sure model
     * config and column config loaded once.
     */
    private void loadConfigFiles(final Properties props) {
        try {
            SourceType sourceType = SourceType.valueOf(props.getProperty(NNConstants.NN_MODELSET_SOURCE_TYPE,
                    SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(NNConstants.SHIFU_NN_MODEL_CONFIG),
                    sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void writeModelWeightsToFileSystem(double[] weights, Path out) {
        FSDataOutputStream fos = null;
        PrintWriter pw = null;
        try {
            fos = FileSystem.get(new Configuration()).create(out);
            LOG.info("Writing results to {}", out);
            if(out != null) {
                pw = new PrintWriter(fos);
                pw.println(Arrays.toString(weights));
            }
        } catch (IOException e) {
            LOG.error("Error in writing output.", e);
        } finally {
            IOUtils.closeStream(pw);
        }
    }

}
