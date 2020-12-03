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
package ml.shifu.shifu.core.dtrain.dt;

import ml.shifu.guagua.master.BasicMasterInterceptor;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.udf.norm.PrecisionType;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.HDFSUtils;
import org.apache.hadoop.fs.FSDataInputStream;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * {@link DTOutput} is used to write the model output and error info to file system.
 */
public class DTOutput extends BasicMasterInterceptor<DTMasterParams, DTWorkerParams> {

    private static final Logger LOG = LoggerFactory.getLogger(DTOutput.class);

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
     * If for random forest running, this is default for such master.
     */
    private boolean isRF = true;

    /**
     * If gradient boost decision tree, for GBDT, each time a tree is trained, next train is trained by gradient label
     * from previous tree.
     */
    private boolean isGBDT = false;

    /**
     * If for grid search, store validation error besides model files.
     */
    private boolean isGsMode;

    /**
     * Valid training parameters including grid search
     */
    private Map<String, Object> validParams;

    /**
     * ColumnConfig list reference
     */
    private List<ColumnConfig> columnConfigList;

    /**
     * input count
     */
    private int inputCount;

    /**
     * Number of trees for both RF and GBDT
     */
    private Integer treeNum;

    /**
     * If k-fold cross validation
     */
    private boolean isKFoldCV;

    /**
     * Model spec precision, by default some threshold and predict are double to ensure model predict precision, change
     * to FLOAT32 or FLOAT16 may impact final predict precision but save some space. As already compressed based on
     * reduce unnessecary fields, category encoding and gzip compression, not too much space saving if FLOAT32 or
     * FLOAT16, FLOAT16 reuses the same float type to store model spec, size of model spec could be the same as FLOAT32.
     */
    private PrecisionType msPt;

    /**
     * Binary model path: bmodels/mode0.gbt.float32 ...
     */
    private Path bModel;

    @Override
    public void preApplication(MasterContext<DTMasterParams, DTWorkerParams> context) {
        init(context);
    }

    @Override
    public void postIteration(final MasterContext<DTMasterParams, DTWorkerParams> context) {
        long start = System.currentTimeMillis();
        // save tmp to hdfs according to raw trainer logic
        final int tmpModelFactor = DTrainUtils.tmpModelFactor(context.getTotalIteration());
        final int currentIteration = context.getCurrentIteration();
        final int totalIteration = context.getTotalIteration();
        final boolean isHalt = context.getMasterResult().isHalt();

        if(isRF) {
            if(currentIteration % (tmpModelFactor * 2) == 0) {
                // save tmp models
                tmpRFModelSave(context, currentIteration, totalIteration, isHalt);
            }
        } else if(isGBDT) {
            // for gbdt, only store trees are all built well
            if(this.treeNum >= 10 && context.getMasterResult().isSwitchToNextTree()
                    && (context.getMasterResult().getTmpTrees().size() - 1) % (this.treeNum / 10) == 0) {
                tmpGBDTModelSave(context, currentIteration, totalIteration, isHalt);
            }
        }

        updateProgressLog(context);
        LOG.debug("DT output post iteration time is {}ms", (System.currentTimeMillis() - start));
    }

    private void tmpGBDTModelSave(final MasterContext<DTMasterParams, DTWorkerParams> context,
            final int currentIteration, final int totalIteration, final boolean isHalt) {
        final List<TreeNode> trees = context.getMasterResult().getTmpTrees();
        if(trees.size() > 1) {
            Thread tmpModelPersistThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    List<TreeNode> subTrees = trees.subList(0, trees.size() - 1);
                    // save model results for continue model training, if current job is failed, then next
                    // running we can start from this point to save time.
                    // another case for master recovery, if master is failed, read such checkpoint model
                    Path out = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));

                    // if current iteration is the last iteration, or it is halted by early stop condition, no
                    // need to save checkpoint model here as it is replicated with postApplicaiton.
                    // There is issue here if saving the same model in this thread and another thread in
                    // postApplication, sometimes this conflict will cause model writing failed.
                    int subTreesSize = subTrees.size();
                    if(!isHalt && currentIteration != totalIteration) {
                        Path tmpModelPath = getTmpModelPath(subTreesSize);
                        writeModelToFileSystem(subTrees, out, false);

                        // in such case tmp model is final model, just copy to tmp models
                        LOG.info("Copy checkpointed model to tmp folder: {}", tmpModelPath.toString());
                        try {
                            DataOutputStream outputStream = new DataOutputStream(
                                    new GZIPOutputStream(HDFSUtils.getFS(tmpModelPath).create(tmpModelPath)));
                            FSDataInputStream inputStream = HDFSUtils.getFS(out).open(out);
                            DataInputStream dis = new DataInputStream(new GZIPInputStream(inputStream));
                            IOUtils.copyBytes(dis, outputStream, HDFSUtils.getConf());
                        } catch (IOException e) {
                            LOG.warn("Error in copy models to tmp", e);
                        }
                    } else {
                        // last one is newest one with only ROOT node, should be excluded
                        saveTmpModelToHDFS(subTreesSize, subTrees);
                    }
                }
            }, "saveTmpModelToHDFS thread");
            tmpModelPersistThread.setDaemon(true);
            tmpModelPersistThread.start();
        }
    }

    private void tmpRFModelSave(final MasterContext<DTMasterParams, DTWorkerParams> context, final int currentIteration,
            final int totalIteration, final boolean isHalt) {
        Thread tmpModelPersistThread = new Thread(new Runnable() {
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
                    writeModelToFileSystem(context.getMasterResult().getTrees(), out, false);

                    // in such case tmp model is final model, just copy to tmp models
                    LOG.info("Copy checkpointed model to tmp folder: {}", tmpModelPath.toString());
                    try {
                        DataOutputStream outputStream = new DataOutputStream(
                                new GZIPOutputStream(HDFSUtils.getFS(tmpModelPath).create(tmpModelPath)));
                        FSDataInputStream inputStream = HDFSUtils.getFS(out).open(out);
                        DataInputStream dis = new DataInputStream(new GZIPInputStream(inputStream));
                        IOUtils.copyBytes(dis, outputStream, HDFSUtils.getConf());
                    } catch (IOException e) {
                        LOG.warn("Error in copy models to tmp", e);
                    }
                } else {
                    // last one only save tmp models
                    saveTmpModelToHDFS(currentIteration, context.getMasterResult().getTrees());
                }
            }
        }, "saveTmpModelToHDFS thread");
        tmpModelPersistThread.setDaemon(true);
        tmpModelPersistThread.start();
    }

    private void updateProgressLog(final MasterContext<DTMasterParams, DTWorkerParams> context) {
        int currentIteration = context.getCurrentIteration();
        if(context.isFirstIteration()) {
            // first iteration is used for training preparation, no need update progress
            return;
        }
        double trainError = context.getMasterResult().getTrainError() / context.getMasterResult().getTrainCount();
        double validationError = context.getMasterResult().getValidationCount() == 0d ? 0d
                : context.getMasterResult().getValidationError() / context.getMasterResult().getValidationCount();
        String info = "";
        if(this.isGBDT) {
            info = buildGBDTProgressInfo(context, currentIteration, trainError, validationError);
        } else if(this.isRF) {
            info = buildRFProgressInfo(context, currentIteration, trainError, validationError);
        }

        if(info.length() > 0) {
            try {
                LOG.debug("Writing progress results to {} {}", context.getCurrentIteration(), info.toString());
                this.progressOutput.write(info.getBytes("UTF-8"));
                this.progressOutput.flush();
                this.progressOutput.hflush();
            } catch (IOException e) {
                LOG.error("Error in write progress log:", e);
            }
        }
    }

    private String buildRFProgressInfo(final MasterContext<DTMasterParams, DTWorkerParams> context,
            int currentIteration, double trainError, double validationError) {
        String info = "";
        if(trainError != 0d) {
            List<Integer> treeDepth = context.getMasterResult().getTreeDepth();
            if(treeDepth.size() == 0) {
                info = new StringBuilder(200).append("Trainer ").append(this.trainerId).append(" Iteration #")
                        .append(currentIteration - 1).append(" Training Error: ")
                        .append(trainError == 0d ? "N/A" : String.format("%.10f", trainError))
                        .append(" Validation Error: ")
                        .append(validationError == 0d ? "N/A" : String.format("%.10f", validationError)).append("\n")
                        .toString();
            } else {
                info = new StringBuilder(200).append("Trainer ").append(this.trainerId).append(" Iteration #")
                        .append(currentIteration - 1).append(" Training Error: ")
                        .append(trainError == 0d ? "N/A" : String.format("%.10f", trainError))
                        .append(" Validation Error: ")
                        .append(validationError == 0d ? "N/A" : String.format("%.10f", validationError))
                        .append("; will work on depth ").append(toListString(treeDepth)).append(". \n").toString();
            }
        }
        return info;
    }

    private String buildGBDTProgressInfo(final MasterContext<DTMasterParams, DTWorkerParams> context,
            int currentIteration, double trainError, double validationError) {
        String info;
        int treeSize = 0;
        if(context.getMasterResult().isSwitchToNextTree() || context.getMasterResult().isHalt()) {
            treeSize = context.getMasterResult().isSwitchToNextTree()
                    ? (context.getMasterResult().getTmpTrees().size() - 1)
                    : (context.getMasterResult().getTmpTrees().size());
            info = new StringBuilder(200).append("Trainer ").append(this.trainerId).append(" Iteration #")
                    .append(currentIteration - 1).append(" Training Error: ")
                    .append((Double.isNaN(trainError) || trainError == 0d) ? "N/A" : String.format("%.10f", trainError))
                    .append(" Validation Error: ")
                    .append(validationError == 0d ? "N/A" : String.format("%.10f", validationError)).append("; Tree ")
                    .append(treeSize).append(" is finished. \n").toString();
        } else {
            int nextDepth = context.getMasterResult().getTreeDepth().get(0);
            info = new StringBuilder(200).append("Trainer ").append(this.trainerId).append(" Iteration #")
                    .append(currentIteration - 1).append(" Training Error: ")
                    .append((Double.isNaN(trainError) || trainError == 0d) ? "N/A" : String.format("%.10f", trainError))
                    .append(" Validation Error: ")
                    .append(validationError == 0d ? "N/A" : String.format("%.10f", validationError))
                    .append("; will work on depth ").append(nextDepth).append(". \n").toString();
        }
        return info;
    }

    /**
     * Show -1 as N/A which means not work on such iteration
     */
    private String toListString(List<Integer> list) {
        Iterator<Integer> i = list.iterator();
        if(!i.hasNext()) {
            return "[]";
        }

        StringBuilder sb = new StringBuilder();
        sb.append('[');
        for(;;) {
            Integer e = i.next();
            sb.append(e == null || e == -1 ? "N/A" : e);
            if(!i.hasNext()) {
                return sb.append(']').toString();
            }
            sb.append(", ");
        }
    }

    @Override
    public void postApplication(MasterContext<DTMasterParams, DTWorkerParams> context) {
        List<TreeNode> trees = context.getMasterResult().getTrees();
        if(this.isGBDT) {
            trees = context.getMasterResult().getTmpTrees();
        }
        Path out = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
        writeModelToFileSystem(trees, out, true);
        if(this.isGsMode || this.isKFoldCV) {
            Path valErrOutput = new Path(context.getProps().getProperty(CommonConstants.GS_VALIDATION_ERROR));
            writeValErrorToFileSystem(
                    context.getMasterResult().getValidationError() / context.getMasterResult().getValidationCount(),
                    valErrOutput);
        }
        IOUtils.closeStream(this.progressOutput);
    }

    private void writeValErrorToFileSystem(double valError, Path out) {
        FSDataOutputStream fos = null;
        try {
            fos = HDFSUtils.getFS(out).create(out);
            LOG.info("Writing valerror to {}", out);
            fos.write((valError + "").getBytes("UTF-8"));
        } catch (IOException e) {
            LOG.error("Error in writing output.", e);
        } finally {
            IOUtils.closeStream(fos);
        }
    }

    private void writeModelToFileSystem(List<TreeNode> trees, Path out, boolean isFinal) {
        List<List<TreeNode>> baggingTrees = new ArrayList<List<TreeNode>>();
        baggingTrees.add(trees);
        try {
            BinaryDTSerializer.save(modelConfig, columnConfigList, baggingTrees,
                    this.validParams.get("Loss").toString(), inputCount, HDFSUtils.getFS(out), out);
            if(isFinal) {
                if(this.msPt != null) {
                    switch(this.msPt) {
                        case FLOAT32:
                        case FLOAT16:
                            Path msOut = new Path(this.bModel.toString() + "." + this.msPt.toString().toLowerCase());
                            BinaryDTSerializer.save(modelConfig, columnConfigList, baggingTrees,
                                    this.validParams.get("Loss").toString(), inputCount, HDFSUtils.getFS(msOut),
                                    msOut, this.msPt);
                            break;
                        default:
                            throw new UnsupportedOperationException("Unsupported model spec precision type.");
                    }
                }
            }
        } catch (IOException e) {
            LOG.error("Error in writing model", e);
        }
    }

    /**
     * Save tmp model to HDFS.
     */
    private void saveTmpModelToHDFS(int iteration, List<TreeNode> trees) {
        Path out = getTmpModelPath(iteration);
        writeModelToFileSystem(trees, out, false);
    }

    private Path getTmpModelPath(int iteration) {
        return new Path(DTrainUtils.getTmpModelName(this.tmpModelsFolder, this.trainerId, iteration,
                modelConfig.getTrain().getAlgorithm().toLowerCase()));
    }

    private void init(MasterContext<DTMasterParams, DTWorkerParams> context) {
        if(isInit.compareAndSet(false, true)) {
            loadConfigFiles(context.getProps());
            this.trainerId = context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID);
            GridSearch gs = new GridSearch(modelConfig.getTrain().getParams(),
                    modelConfig.getTrain().getGridConfigFileContent());
            this.isGsMode = gs.hasHyperParam();

            this.validParams = modelConfig.getParams();
            if(isGsMode) {
                this.validParams = gs.getParams(Integer.parseInt(trainerId));
            }

            Integer kCrossValidation = this.modelConfig.getTrain().getNumKFold();
            this.isKFoldCV = kCrossValidation != null && kCrossValidation > 0;

            this.tmpModelsFolder = context.getProps().getProperty(CommonConstants.SHIFU_TMP_MODELS_FOLDER);
            this.isRF = ALGORITHM.RF.toString().equalsIgnoreCase(modelConfig.getAlgorithm());
            this.isGBDT = ALGORITHM.GBT.toString().equalsIgnoreCase(modelConfig.getAlgorithm());
            int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
            // numerical + categorical = # of all input
            this.inputCount = inputOutputIndex[0] + inputOutputIndex[1];
            try {
                Path progressLog = new Path(context.getProps().getProperty(CommonConstants.SHIFU_DTRAIN_PROGRESS_FILE));
                // if the progressLog already exists, that because the master failed, and fail-over
                // we need to append the log, so that client console can get refreshed. Or console will appear stuck.
                if(ShifuFileUtils.isFileExists(progressLog, SourceType.HDFS)) {
                    this.progressOutput = HDFSUtils.getFS(progressLog).append(progressLog);
                } else {
                    this.progressOutput = HDFSUtils.getFS(progressLog).create(progressLog);
                }
            } catch (IOException e) {
                LOG.error("Error in create progress log:", e);
            }
            this.treeNum = Integer.valueOf(validParams.get("TreeNum").toString());;

            try {
                this.msPt = PrecisionType.of(context.getProps().getProperty(Constants.SHIFU_MODELSPEC_PRECISION_TYPE));
            } catch (Exception e) {
                this.msPt = null;
            }
            this.bModel = new Path(context.getProps().getProperty(Constants.SHIFU_BINARY_MODEL_PATH));
            LOG.info("Model spec precision: " + this.msPt);
        }
    }

    /**
     * Load all configurations for modelConfig and columnConfigList from source type. Use null check to make sure model
     * config and column config loaded once.
     */
    private void loadConfigFiles(final Properties props) {
        try {
            SourceType sourceType = SourceType
                    .valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE, SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
            this.columnConfigList = CommonUtils
                    .loadColumnConfigList(props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
