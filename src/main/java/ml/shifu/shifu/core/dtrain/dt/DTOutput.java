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

import java.io.DataOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.zip.GZIPOutputStream;

import ml.shifu.guagua.master.BasicMasterInterceptor;
import ml.shifu.guagua.master.MasterContext;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.core.dtrain.DTrainUtils;
import ml.shifu.shifu.core.dtrain.gs.GridSearch;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.collections.CollectionUtils;
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

    @Override
    public void preApplication(MasterContext<DTMasterParams, DTWorkerParams> context) {
        init(context);
    }

    @Override
    public void postIteration(final MasterContext<DTMasterParams, DTWorkerParams> context) {
        long start = System.currentTimeMillis();
        // save tmp to hdfs according to raw trainer logic
        final int tmpModelFactor = DTrainUtils.tmpModelFactor(context.getTotalIteration());
        if(isRF) {
            if(context.getCurrentIteration() % (tmpModelFactor * 2) == 0) {
                Thread tmpModelPersistThread = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        // save model results for continue model training, if current job is failed, then next running
                        // we can start from this point to save time.
                        // another case for master recovery, if master is failed, read such checkpoint model
                        Path out = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
                        writeModelToFileSystem(context.getMasterResult().getTrees(), out);

                        saveTmpModelToHDFS(context.getCurrentIteration(), context.getMasterResult().getTrees());
                    }
                }, "saveTmpNNToHDFS thread");
                tmpModelPersistThread.setDaemon(true);
                tmpModelPersistThread.start();
            }
        } else if(isGBDT) {
            // for gbdt, only store trees are all built well
            if(this.treeNum >= 10 && context.getMasterResult().isSwitchToNextTree()
                    && (context.getMasterResult().getTmpTrees().size() - 1) % (this.treeNum / 10) == 0) {
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
                            writeModelToFileSystem(subTrees, out);
                            // last one is newest one with only ROOT node, should be excluded
                            saveTmpModelToHDFS(subTrees.size(), subTrees);
                        }
                    }, "saveTmpNNToHDFS thread");
                    tmpModelPersistThread.setDaemon(true);
                    tmpModelPersistThread.start();
                }
            }
        }

        updateProgressLog(context);
        LOG.info("DT output post iteration time is {}ms", (System.currentTimeMillis() - start));
    }

    @SuppressWarnings("deprecation")
    private void updateProgressLog(final MasterContext<DTMasterParams, DTWorkerParams> context) {
        int currentIteration = context.getCurrentIteration();
        if(context.isFirstIteration()) {
            // first iteration is used for training preparation
            return;
        }
        double trainError = context.getMasterResult().getTrainError() / context.getMasterResult().getTrainCount();
        double validationError = context.getMasterResult().getValidationCount() == 0d ? 0d : context.getMasterResult()
                .getValidationError() / context.getMasterResult().getValidationCount();
        String info = "";
        if(this.isGBDT) {
            int treeSize = 0;
            if(context.getMasterResult().isSwitchToNextTree() || context.getMasterResult().isHalt()) {
                treeSize = context.getMasterResult().isSwitchToNextTree() ? (context.getMasterResult().getTmpTrees()
                        .size() - 1) : (context.getMasterResult().getTmpTrees().size());
                info = new StringBuilder(200)
                        .append("Trainer ")
                        .append(this.trainerId)
                        .append(" Iteration #")
                        .append(currentIteration - 1)
                        .append(" Train Error: ")
                        .append((Double.isNaN(trainError) || trainError == 0d) ? "N/A" : String.format("%.10f",
                                trainError)).append(" Validation Error: ")
                        .append(validationError == 0d ? "N/A" : String.format("%.10f", validationError))
                        .append("; Tree ").append(treeSize).append(" is finished. \n").toString();
            } else {
                int nextDepth = context.getMasterResult().getTreeDepth().get(0);
                info = new StringBuilder(200)
                        .append("Trainer ")
                        .append(this.trainerId)
                        .append(" Iteration #")
                        .append(currentIteration - 1)
                        .append(" Train Error: ")
                        .append((Double.isNaN(trainError) || trainError == 0d) ? "N/A" : String.format("%.10f",
                                trainError)).append(" Validation Error: ")
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
        if(this.isGBDT) {
            trees = context.getMasterResult().getTmpTrees();
        }
        LOG.debug("final trees", trees.toString());
        Path out = new Path(context.getProps().getProperty(CommonConstants.GUAGUA_OUTPUT));
        writeModelToFileSystem(trees, out);
        if(this.isGsMode) {
            Path valErrOutput = new Path(context.getProps().getProperty(CommonConstants.GS_VALIDATION_ERROR));
            writeValErrorToFileSystem(context.getMasterResult().getValidationError()
                    / context.getMasterResult().getValidationCount(), valErrOutput);
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

    private void writeModelToFileSystem(List<TreeNode> trees, Path out) {
        DataOutputStream fos = null;

        try {
             fos = new DataOutputStream(new GZIPOutputStream(FileSystem.get(new Configuration()).create(out)));
//            fos = new DataOutputStream(FileSystem.get(new Configuration()).create(out));
            LOG.info("Writing  {} trees to {}.", trees.size(), out);
            // version
            fos.writeInt(CommonConstants.TREE_FORMAT_VERSION);
            fos.writeUTF(modelConfig.getAlgorithm());
            fos.writeUTF(this.validParams.get("Loss").toString());
            fos.writeBoolean(this.modelConfig.isClassification());
            fos.writeBoolean(this.modelConfig.getTrain().isOneVsAll());
            fos.writeInt(this.inputCount);

            Map<Integer, String> columnIndexNameMapping = new HashMap<Integer, String>();
            Map<Integer, List<String>> columnIndexCategoricalListMapping = new HashMap<Integer, List<String>>();
            Map<Integer, Double> numericalMeanMapping = new HashMap<Integer, Double>();
            for(ColumnConfig columnConfig: this.columnConfigList) {
                columnIndexNameMapping.put(columnConfig.getColumnNum(), columnConfig.getColumnName());
                if(columnConfig.isCategorical() && CollectionUtils.isNotEmpty(columnConfig.getBinCategory())) {
                    columnIndexCategoricalListMapping.put(columnConfig.getColumnNum(), columnConfig.getBinCategory());
                }

                if(columnConfig.isNumerical() && columnConfig.getMean() != null) {
                    numericalMeanMapping.put(columnConfig.getColumnNum(), columnConfig.getMean());
                }
            }

            // serialize numericalMeanMapping
            fos.writeInt(numericalMeanMapping.size());
            for(Entry<Integer, Double> entry: numericalMeanMapping.entrySet()) {
                fos.writeInt(entry.getKey());
                // for some feature, it is null mean value, it is not selected, just set to 0d to avoid NPE
                fos.writeDouble(entry.getValue() == null ? 0d : entry.getValue());
            }
            // serialize columnIndexNameMapping
            fos.writeInt(columnIndexNameMapping.size());
            for(Entry<Integer, String> entry: columnIndexNameMapping.entrySet()) {
                fos.writeInt(entry.getKey());
                fos.writeUTF(entry.getValue());
            }
            // serialize columnIndexCategoricalListMapping
            fos.writeInt(columnIndexCategoricalListMapping.size());
            for(Entry<Integer, List<String>> entry: columnIndexCategoricalListMapping.entrySet()) {
                List<String> categories = entry.getValue();
                if(categories != null) {
                    fos.writeInt(entry.getKey());
                    fos.writeInt(categories.size());
                    for(String category: categories) {
                        fos.writeUTF(category);
                    }
                }
            }

            Map<Integer, Integer> columnMapping = getColumnMapping();
            fos.writeInt(columnMapping.size());
            for(Entry<Integer, Integer> entry: columnMapping.entrySet()) {
                fos.writeInt(entry.getKey());
                fos.writeInt(entry.getValue());
            }

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

    private Map<Integer, Integer> getColumnMapping() {
        Map<Integer, Integer> columnMapping = new HashMap<Integer, Integer>(columnConfigList.size(), 1f);
        int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(columnConfigList);
        boolean isAfterVarSelect = inputOutputIndex[3] == 1 ? true : false;
        int index = 0;
        for(int i = 0; i < columnConfigList.size(); i++) {
            ColumnConfig columnConfig = columnConfigList.get(i);
            if(!isAfterVarSelect) {
                if(!columnConfig.isMeta() && !columnConfig.isTarget() && CommonUtils.isGoodCandidate(columnConfig)) {
                    columnMapping.put(columnConfig.getColumnNum(), index);
                    index += 1;
                }
            } else {
                if(columnConfig != null && !columnConfig.isMeta() && !columnConfig.isTarget()
                        && columnConfig.isFinalSelect()) {
                    columnMapping.put(columnConfig.getColumnNum(), index);
                    index += 1;
                }
            }
        }
        return columnMapping;
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
            GridSearch gs = new GridSearch(modelConfig.getTrain().getParams());
            this.isGsMode = gs.hasHyperParam();

            this.validParams = modelConfig.getParams();
            if(isGsMode) {
                this.validParams = gs.getParams(Integer.parseInt(trainerId));
            }

            this.tmpModelsFolder = context.getProps().getProperty(CommonConstants.SHIFU_TMP_MODELS_FOLDER);
            this.isRF = ALGORITHM.RF.toString().equalsIgnoreCase(modelConfig.getAlgorithm());
            this.isGBDT = ALGORITHM.GBT.toString().equalsIgnoreCase(modelConfig.getAlgorithm());
            int[] inputOutputIndex = DTrainUtils.getNumericAndCategoricalInputAndOutputCounts(this.columnConfigList);
            // numerical + categorical = # of all input
            this.inputCount = inputOutputIndex[0] + inputOutputIndex[1];
            try {
                Path progressLog = new Path(context.getProps().getProperty(CommonConstants.SHIFU_DTRAIN_PROGRESS_FILE));
                this.progressOutput = FileSystem.get(new Configuration()).create(progressLog);
            } catch (IOException e) {
                LOG.error("Error in create progress log:", e);
            }
            this.treeNum = Integer.valueOf(validParams.get("TreeNum").toString());;
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
            this.columnConfigList = CommonUtils.loadColumnConfigList(
                    props.getProperty(CommonConstants.SHIFU_COLUMN_CONFIG), sourceType);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

}
