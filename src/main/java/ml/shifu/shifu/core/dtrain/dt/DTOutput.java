/**
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

import java.io.IOException;
import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;

import ml.shifu.guagua.master.BasicMasterInterceptor;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FSDataOutputStream;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.IOUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * {@link DTOutput} is used to write the model output and error info to file system.
 */
public class DTOutput extends BasicMasterInterceptor<DTMasterParams, DTWorkerParams> {

    private static final Logger LOG = LoggerFactory.getLogger(DTOutput.class);

    /**
     * Model Config read from HDFS
     */
    private ModelConfig modelConfig;

    private String trainerId;

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

    @Override
    public void preApplication(MasterContext<DTMasterParams, DTWorkerParams> context) {
        init(context);
    }

    @Override
    public void postIteration(final MasterContext<DTMasterParams, DTWorkerParams> context) {
        // save tmp to hdfs according to raw trainer logic
        final int tmpModelFactor = DTrainUtils.tmpModelFactor(context.getTotalIteration());
        if(context.getCurrentIteration() % tmpModelFactor == 0) {
            Thread tmpModelPersistThread = new Thread(new Runnable() {
                @Override
                public void run() {
                    saveTmpModelToHDFS(context.getCurrentIteration(), context.getMasterResult().getTrees());
                    // save model results for continue model training, if current job is failed, then next running
                    // we can start from this point to save time.
                    // another case for master recovery, if master is failed, read such checkpoint model
                    Path out = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
                    writeModelToFileSystem(context.getMasterResult().getTrees(), out);
                }
            }, "saveTmpNNToHDFS thread");
            tmpModelPersistThread.setDaemon(true);
            tmpModelPersistThread.start();
        }

        updateProgressLog(context);
    }

    @SuppressWarnings("deprecation")
    private void updateProgressLog(final MasterContext<DTMasterParams, DTWorkerParams> context) {
        int currentIteration = context.getCurrentIteration();
        if(context.isFirstIteration()) {
            // first iteration is used for training preparation
            return;
        }
        double trainError = context.getMasterResult().getTrainError() / context.getMasterResult().getTrainCount();
        double validationError = context.getMasterResult().getValidationCount() == 0L ? 0d : context.getMasterResult()
                .getValidationError() / context.getMasterResult().getValidationCount();
        String info = "";
        if(this.isGBDT) {
            int treeSize = 0;
            if(context.getMasterResult().isSwitchToNextTree() || context.getMasterResult().isHalt()) {
                treeSize = context.getMasterResult().isSwitchToNextTree() ? (context.getMasterResult().getTrees()
                        .size() - 1) : (context.getMasterResult().getTrees().size());
                info = new StringBuilder(200).append("Trainer ").append(this.trainerId).append(" Iteration #")
                        .append(currentIteration - 1).append(" Train Error: ")
                        .append(String.format("%.10f", trainError)).append(" Validation Error: ")
                        .append(validationError == 0d ? "N/A" : String.format("%.10f", validationError))
                        .append("; Tree ").append(treeSize).append(" (starting from 1)  is finished. \n")
                        .toString();
            } else {
                int treeIndex = context.getMasterResult().getTrees().size() - 1;
                int nextDepth = context.getMasterResult().getTreeDepth().get(treeIndex);
                info = new StringBuilder(200).append("Trainer ").append(this.trainerId).append(" Iteration #")
                        .append(currentIteration - 1).append(" Train Error: ")
                        .append(trainError == 0d ? "N/A" : String.format("%.10f", trainError))
                        .append(" Validation Error: ")
                        .append(validationError == 0d ? "N/A" : String.format("%.10f", validationError))
                        .append("; will work on depth ").append(nextDepth).append(" \n").toString();
            }
        }

        if(this.isRF) {
            if(trainError != 0d) {
                List<Integer> treeDepth = context.getMasterResult().getTreeDepth();
                if(treeDepth.size() == 0) {
                    info = new StringBuilder(200).append("Trainer ").append(this.trainerId).append(" Iteration #")
                            .append(currentIteration - 1).append(" Train Error: ")
                            .append(trainError == 0d ? "N/A" : String.format("%.10f", trainError))
                            .append(" Validation Error: ")
                            .append(validationError == 0d ? "N/A" : String.format("%.10f", validationError))
                            .append("\n").toString();
                } else {
                    info = new StringBuilder(200).append("Trainer ").append(this.trainerId).append(" Iteration #")
                            .append(currentIteration - 1).append(" Train Error: ")
                            .append(trainError == 0d ? "N/A" : String.format("%.10f", trainError))
                            .append(" Validation Error: ")
                            .append(validationError == 0d ? "N/A" : String.format("%.10f", validationError))
                            .append("; will work on depth ").append(toListString(treeDepth)).append("\n").toString();
                }
            }
        }

        if(info.length() > 0) {
            try {
                LOG.debug("Writing progress results to {} {}", context.getCurrentIteration(), info.toString());
                this.progressOutput.write(info.getBytes("UTF-8"));
                this.progressOutput.flush();
                this.progressOutput.sync();
            } catch (IOException e) {
                LOG.error("Error in write progress log:", e);
            }
        }
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
        LOG.info("final trees", trees.toString());
        Path out = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
        writeModelToFileSystem(trees, out);
    }

    private void writeModelToFileSystem(List<TreeNode> trees, Path out) {
        FSDataOutputStream fos = null;
        try {
            fos = FileSystem.get(new Configuration()).create(out);
            LOG.info("Writing results to {}", out);
            int treeLength = trees.size();
            fos.writeInt(treeLength);
            for(TreeNode treeNode: trees) {
                treeNode.write(fos);
            }
        } catch (IOException e) {
            LOG.error("Error in writing output.", e);
        } finally {
            IOUtils.closeStream(fos);
        }
    }

    /**
     * Save tmp model to HDFS.
     */
    private void saveTmpModelToHDFS(int iteration, List<TreeNode> trees) {
        Path out = new Path(DTrainUtils.getTmpModelName(this.tmpModelsFolder, this.trainerId, iteration, modelConfig
                .getTrain().getAlgorithm().toLowerCase()));
        writeModelToFileSystem(trees, out);
    }

    private void init(MasterContext<DTMasterParams, DTWorkerParams> context) {
        if(isInit.compareAndSet(false, true)) {
            loadConfigFiles(context.getProps());
            this.trainerId = context.getProps().getProperty(CommonConstants.SHIFU_TRAINER_ID);
            this.tmpModelsFolder = context.getProps().getProperty(CommonConstants.SHIFU_TMP_MODELS_FOLDER);
            this.isRF = ALGORITHM.RF.toString().equalsIgnoreCase(modelConfig.getAlgorithm());
            this.isGBDT = ALGORITHM.GBT.toString().equalsIgnoreCase(modelConfig.getAlgorithm());
        }

        try {
            Path progressLog = new Path(context.getProps().getProperty(CommonConstants.SHIFU_DTRAIN_PROGRESS_FILE));
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
            SourceType sourceType = SourceType.valueOf(props.getProperty(CommonConstants.MODELSET_SOURCE_TYPE,
                    SourceType.HDFS.toString()));
            this.modelConfig = CommonUtils.loadModelConfig(props.getProperty(CommonConstants.SHIFU_MODEL_CONFIG),
                    sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
