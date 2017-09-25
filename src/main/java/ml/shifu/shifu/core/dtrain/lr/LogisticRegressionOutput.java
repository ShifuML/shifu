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
package ml.shifu.shifu.core.dtrain.lr;

import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

import ml.shifu.guagua.master.BasicMasterInterceptor;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
//import ml.shifu.shifu.core.dtrain.nn.NNConstants;
import ml.shifu.shifu.fs.ShifuFileUtils;
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
    private double[] optimizedWeights = null;

    /**
     * Progress output stream which is used to write progress to that HDFS file. Should be closed in
     * {@link #postApplication(MasterContext)}.
     */
    private FSDataOutputStream progressOutput = null;

    /**
     * If current mode is cross validation
     */
    private boolean isKFoldCV;

    /**
     * If current mode is grid search
     */
    private boolean isGsMode;

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
            this.optimizedWeights = context.getMasterResult().getParameters();
        }

        // save tmp to hdfs according to raw trainer logic
        final int tmpModelFactor = DTrainUtils.tmpModelFactor(context.getTotalIteration());
        final int currentIteration = context.getCurrentIteration();
        final double[] parameters = context.getMasterResult().getParameters();
        final int totalIteration = context.getTotalIteration();
        final boolean isHalt = context.getMasterResult().isHalt();
        // currentIteration - 1 because the first iteration is used for sync master models to workers
        if((currentIteration - 1) % tmpModelFactor == 0) {
            Thread tmpNNThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    saveTmpModelToHDFS(currentIteration - 1, parameters);
                    // save model results for continue model training, if current job is failed, then next running
                    // we can start from this point to save time.
                    // another case for master recovery, if master is failed, read such checkpoint model
                    Path out = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));

                    // if current iteration is the last iteration, or it is halted by early stop condition, no
                    // need to save checkpoint model here as it is replicated with postApplicaiton.
                    // There is issue here if saving the same model in this thread and another thread in
                    // postApplication, sometimes this conflict will cause model writing failed.
                    if(!isHalt && currentIteration != totalIteration) {
                        writeModelWeightsToFileSystem(optimizedWeights, out);
                    }
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
                .append(currentIteration - 1).append(" Training Error:")
                .append(context.getMasterResult().getTrainError()).append(" Validation Error:")
                .append(context.getMasterResult().getTestError()).append("\n").toString();
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

        if(optimizedWeights == null) {
            optimizedWeights = context.getMasterResult().getParameters();
        }

        Path out = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
        writeModelWeightsToFileSystem(optimizedWeights, out);
        if(this.isKFoldCV || this.isGsMode) {
            Path valErrOutput = new Path(context.getProps().getProperty(CommonConstants.GS_VALIDATION_ERROR));
            writeValErrorToFileSystem(context.getMasterResult().getTestError(), valErrOutput);
        }
        IOUtils.closeStream(this.progressOutput);
    }

    private void writeValErrorToFileSystem(double valError, Path out) {
        FSDataOutputStream fos = null;
        try {
            fos = FileSystem.get(new Configuration()).create(out);
            LOG.info("Writing valerror to {}", out);
            fos.write((valError + "").getBytes("UTF-8"));
        } catch (IOException e) {
            LOG.error("Error in writing output.", e);
        } finally {
            IOUtils.closeStream(fos);
        }
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
        this.isDry = Boolean.TRUE.toString().equals(context.getProps().getProperty(CommonConstants.SHIFU_DRY_DTRAIN));

        if(this.isDry) {
            return;
        }
        if(isInit.compareAndSet(false, true)) {
            loadConfigFiles(context.getProps());
            this.trainerId = context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID);
            this.tmpModelsFolder = context.getProps().getProperty(CommonConstants.SHIFU_TMP_MODELS_FOLDER);
            Integer kCrossValidation = this.modelConfig.getTrain().getNumKFold();
            if(kCrossValidation != null && kCrossValidation > 0) {
                isKFoldCV = true;
            }

            GridSearch gs = new GridSearch(modelConfig.getTrain().getParams(), modelConfig.getTrain().getGridConfigFileContent());
            this.isGsMode = gs.hasHyperParam();
        }

        try {
            Path progressLog = new Path(context.getProps().getProperty(CommonConstants.SHIFU_DTRAIN_PROGRESS_FILE));
            // if the progressLog already exists, that because the master failed, and fail-over
            // we need to append the log, so that client console can get refreshed. Or console will appear stuck.
            if(ShifuFileUtils.isFileExists(progressLog, SourceType.HDFS)) {
                this.progressOutput = FileSystem.get(new Configuration()).append(progressLog);
            } else {
                this.progressOutput = FileSystem.get(new Configuration()).create(progressLog);
            }
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
            SourceType sourceType = SourceType.valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE,
                    SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    private void writeModelWeightsToFileSystem(double[] weights, Path out) {
        if(weights == null || weights.length <= 0) {
            return;
        }
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
