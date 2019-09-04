/*
 * Copyright [2013-2019] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain.mtl;

import ml.shifu.guagua.master.BasicMasterInterceptor;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.AssertUtils;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * @author haillu
 */
public class MTLOutput extends BasicMasterInterceptor<MTLParams, MTLParams> {
    private static final org.slf4j.Logger LOG = LoggerFactory.getLogger(MTLOutput.class);

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    /**
     * Id for current guagua job, starting from 0, 1, 2
     */
    private String trainerId;

    /**
     * Tmp model folder to save tmp models
     */
    private String tmpModelsFolder;

    /**
     * A flag: whether params initialized.
     */
    private AtomicBoolean isInit = new AtomicBoolean(false);

    /**
     * Progress output stream which is used to write progress to that HDFS file. Should be closed in
     * {@link #postApplication(MasterContext)}.
     */
    private FSDataOutputStream progressOutput = null;

    /**
     * If for grid search, store validation error besides model files.
     */
    private boolean isGsMode;

    /**
     * ColumnConfig lists reference
     */
    private List<List<ColumnConfig>> mtlColumnConfigLists = new ArrayList<>();

    /**
     * If k-fold cross validation
     */
    private boolean isKFoldCV;

    /**
     * Use the same one conf instance
     */
    private Configuration conf;

    @Override
    public void preApplication(MasterContext<MTLParams, MTLParams> context) {
        init(context);
    }

    @Override
    public void postIteration(MasterContext<MTLParams, MTLParams> context) {
        long start = System.currentTimeMillis();
        // save tmp to hdfs according to raw trainer logic
        final int tmpModelFactor = DTrainUtils.tmpModelFactor(context.getTotalIteration());
        final int currentIteration = context.getCurrentIteration();
        final int totalIteration = context.getTotalIteration();
        final boolean isHalt = context.getMasterResult().isHalt();

        if(currentIteration % tmpModelFactor == 0) {
            Thread modePersistThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    // save model results for continue model training, if current job is failed, then next
                    // running we can start from this point to save time. Another case for master recovery, if master is
                    // failed, read such checkpoint model
                    Path out = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
                    // if current iteration is the last iteration, or it is halted by early stop condition, no
                    // need to save checkpoint model here as it is replicated with postApplicaiton.
                    // There is issue here if saving the same model in this thread and another thread in
                    // postApplication, sometimes this conflict will cause model writing failed.
                    if(!isHalt && currentIteration != totalIteration) {
                        Path tmpModelPath = getTmpModelPath(currentIteration);
                        writeModelToFileSystem(context.getMasterResult(), out);
                        // a bug in new version to write last model, wait 1s for model flush hdfs successfully.
                        try {
                            Thread.sleep(100);
                        } catch (InterruptedException e1) {
                            Thread.currentThread().interrupt();
                        }
                        // in such case tmp model is final model, just copy to tmp models
                        LOG.info("Copy checkpointed model to tmp folder: {}", tmpModelPath.toString());
                        try {
                            DataOutputStream outputStream = new DataOutputStream(
                                    new GZIPOutputStream(FileSystem.get(MTLOutput.this.conf).create(tmpModelPath)));
                            FSDataInputStream inputStream = FileSystem.get(MTLOutput.this.conf).open(out);
                            DataInputStream dis = new DataInputStream(new GZIPInputStream(inputStream));
                            IOUtils.copyBytes(dis, outputStream, MTLOutput.this.conf);
                        } catch (IOException e) {
                            LOG.warn("Error in copy models to tmp", e);
                        }
                    } else {
                        // last one only save tmp models
                        saveTmpModelToHDFS(currentIteration, context.getMasterResult());
                    }
                }
            }, "SaveTmpModelToHDFS Thread");
            modePersistThread.setDaemon(true);
            modePersistThread.start();
        }

        updateProgressLog(context);
        LOG.debug("Write output model in post iteration time is {}ms", (System.currentTimeMillis() - start));
    }

    private void updateProgressLog(final MasterContext<MTLParams, MTLParams> context) {
        int currentIteration = context.getCurrentIteration();
        if(context.isFirstIteration()) {
            // first iteration is used for training preparation
            return;
        }

        // compute trainErrors
        double[] trainErrors = context.getMasterResult().getTrainErrors();
        double trainSize = context.getMasterResult().getTrainSize();
        for(int i = 0; i < trainErrors.length; i++) {
            trainErrors[i] /= trainSize;
        }

        // compute validationErrors
        double[] validationErrors = context.getMasterResult().getValidationErrors();
        double validationSize = context.getMasterResult().getValidationSize();
        for(int i = 0; i < validationErrors.length; i++) {
            validationErrors[i] /= validationSize;
        }

        // write the log of trainErrors and validationErrors for different tasks.
        AssertUtils.assertDoubleArrayNotNullAndLengthEqual(trainErrors, validationErrors);
        int taskNumber = trainErrors.length;
        ArrayList<String> infos = new ArrayList<>();
        for(int i = 0; i < taskNumber; i++) {
            if(trainErrors[i] != 0) {
                String info = new StringBuilder(200).append("Trainer ").append(this.trainerId).append(" Iteration #")
                        .append(currentIteration - 1).append(" Training Error: ")
                        .append(trainErrors[i] == 0d ? "N/A" : String.format("%.10f", trainErrors[i]))
                        .append(" Validation Error: ")
                        .append(validationErrors[i] == 0d ? "N/A" : String.format("%.10f", validationErrors[i]))
                        .append("\n").toString();
                infos.add(info);
            }
        }

        if(infos.size() > 0) {
            try {
                for(String info: infos) {
                    LOG.debug("Writing progress results to {} {}", context.getCurrentIteration(), info);
                    this.progressOutput.write(info.getBytes("UTF-8"));
                }
                this.progressOutput.flush();
                this.progressOutput.sync();
            } catch (IOException e) {
                LOG.error("Error in write progress log", e);
            }
        }

    }

    /**
     * Save tmp model to HDFS.
     */
    private void saveTmpModelToHDFS(int iteration, MTLParams params) {
        Path out = getTmpModelPath(iteration);
        writeModelToFileSystem(params, out);
    }

    private Path getTmpModelPath(int iteration) {
        return new Path(DTrainUtils.getTmpModelName(this.tmpModelsFolder, this.trainerId, iteration,
                modelConfig.getTrain().getAlgorithm().toLowerCase()));
    }

    @Override
    public void postApplication(MasterContext<MTLParams, MTLParams> context) {
        Path out = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
        writeModelToFileSystem(context.getMasterResult(), out);
        if(this.isGsMode || this.isKFoldCV) {
            Path valErrOutput = new Path(context.getProps().getProperty(CommonConstants.GS_VALIDATION_ERROR));
            double[] valErrs = context.getMasterResult().getValidationErrors();
            double valSize = context.getMasterResult().getValidationSize();
            for(int i = 0; i < valErrs.length; i++) {
                valErrs[i] /= valSize;
            }
            writeValErrorToFileSystem(valErrs, valErrOutput);
        }
        IOUtils.closeStream(this.progressOutput);
    }

    private void writeValErrorToFileSystem(double[] valErrors, Path out) {
        FSDataOutputStream fos = null;
        try {
            fos = FileSystem.get(new Configuration()).create(out);
            LOG.info("Writing valErrors to {}", out);
            for(int i = 0; i < valErrors.length; i++) {
                fos.write((valErrors[i] + "").getBytes("UTF-8"));
            }
        } catch (IOException e) {
            LOG.error("Error in writing output.", e);
        } finally {
            IOUtils.closeStream(fos);
        }
    }

    private void writeModelToFileSystem(MTLParams params, Path out) {
        try {
            BinaryMTLSerializer.save(this.modelConfig, this.mtlColumnConfigLists, params.getMtl(),
                    FileSystem.get(new Configuration()), out);
        } catch (IOException e) {
            LOG.error("Error in writing MultiTaskLearning model", e);
        }
    }

    private void init(MasterContext<MTLParams, MTLParams> context) {
        if(isInit.compareAndSet(false, true)) {
            this.conf = new Configuration();
            loadConfigs(context.getProps());
            this.trainerId = context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID);
            GridSearch gs = new GridSearch(modelConfig.getTrain().getParams(),
                    modelConfig.getTrain().getGridConfigFileContent());
            this.isGsMode = gs.hasHyperParam();

            Integer kCrossValidation = this.modelConfig.getTrain().getNumKFold();
            if(kCrossValidation != null && kCrossValidation > 0) {
                isKFoldCV = true;
            }

            this.tmpModelsFolder = context.getProps().getProperty(CommonConstants.SHIFU_TMP_MODELS_FOLDER);
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

    }

    /**
     * Load all configurations for modelConfig and columnConfigLists from source type. Use null check to make sure model
     * config and column config loaded once.
     */
    private void loadConfigs(Properties props) {
        try {
            SourceType sourceType = SourceType
                    .valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);

            // build mtlColumnConfigLists.
            List<String> tagColumns = this.modelConfig.getMultiTaskTargetColumnNames();
            PathFinder pf = new PathFinder(this.modelConfig);
            int taskNumber = tagColumns.size();
            for(int i = 0; i < taskNumber; i++) {
                List<ColumnConfig> ccs;
                ccs = CommonUtils.loadColumnConfigList(pf.getMTLColumnConfigPath(sourceType, i), sourceType);
                // for local test
                // ccs = CommonUtils.loadColumnConfigList(
                // "/C:/Users/haillu/Documents/gitRepo/shifu/target/test-classes/model/MultiTaskNN/mtl/ColumnConfig.json."+i,
                // sourceType);

                mtlColumnConfigLists.add(ccs);
            }
        } catch (IOException e) {
            throw new RuntimeException();
        }
    }

}
