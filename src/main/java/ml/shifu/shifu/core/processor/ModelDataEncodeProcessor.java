package ml.shifu.shifu.core.processor;

import com.google.common.io.Files;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.Path;
import org.apache.pig.tools.pigstats.JobStats;
import org.apache.pig.tools.pigstats.PigStats;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.*;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

public class ModelDataEncodeProcessor extends BasicModelProcessor {

    private static final Logger LOG = LoggerFactory.getLogger(ModelDataEncodeProcessor.class);
    public static final String ENCODE_DATA_SET = "ENCODE_DATA_SET";
    public static final String ENCODE_REF_MODEL = "ENCODE_REF_MODEL";
    public static final String TRAINING_DATA_SET = "TDS";

    private IndependentTreeModel treeModel;

    public ModelDataEncodeProcessor(Map<String, Object> params) {
        this.params = params;

        // load Tree model
        InputStream inputStream = null;
        try {
            loadModelConfig();
            this.pathFinder = new PathFinder(this.modelConfig);

            Path modelPath = new Path(this.pathFinder.getModelsPath(SourceType.LOCAL), getModelName(0));
            LOG.info("loading model from - {}", modelPath.toString());
            inputStream = ShifuFileUtils.getInputStream(modelPath, SourceType.LOCAL);
            this.treeModel = IndependentTreeModel.loadFromStream(inputStream);
        } catch (IOException e) {
            throw new RuntimeException("Fail to load GBT model", e);
        } finally {
            IOUtils.closeQuietly(inputStream);
        }
    }

    public int run() {
        LOG.info("Step Start: encode");
        int status = 0;

        try {
            setUp(ModelStep.ENCODE);
            syncDataToHdfs(modelConfig.getDataSet().getSource());

            String encodeRefModel = getEncodeRefModel();
            if(StringUtils.isNotBlank(encodeRefModel) && !ShifuFileUtils
                    .isFileExists(encodeRefModel, RawSourceData.SourceType.LOCAL)) {
                // user set the encode reference model, but it doesn't exist yet
                File tmpDir = Files.createTempDir();
                ShifuFileUtils.copy(".", tmpDir.getPath(), RawSourceData.SourceType.LOCAL);
                ShifuFileUtils.copy(tmpDir.getPath(), encodeRefModel, RawSourceData.SourceType.LOCAL);
                FileUtils.deleteDirectory(tmpDir);

                updateModel(encodeRefModel);
            }

            String encodeDataSet = getEncodeDataSet();
            if(TRAINING_DATA_SET.equals(encodeDataSet)) {
                // run data encode for training data set
                status = encodeModelData(null);
                updateTrainEncodeDataPath(status, encodeRefModel);
            } else if(StringUtils.isNotBlank(encodeDataSet)) {
                // run data encode for evaluation data set
                List<EvalConfig> evalConfigList = getEvalConfigList(encodeDataSet);
                if(CollectionUtils.isEmpty(evalConfigList)) {
                    LOG.error("The EvalSet(s) specified - {} doesn't exits. Please check!", encodeDataSet);
                    status = 1;
                } else {
                    for(EvalConfig evalConfig : evalConfigList) {
                        status = encodeModelData(evalConfig);
                        updateEvalEncodeDataPath(status, encodeRefModel, evalConfig);
                        if(status != 0) {
                            break;
                        }
                    }
                }
            } else { // run data encode for all data set
                // run data encode for training data set
                status = encodeModelData(null);
                updateTrainEncodeDataPath(status, encodeRefModel);

                // run data encode for all evaluation data set
                List<EvalConfig> evalConfigList = this.modelConfig.getEvals();
                if(status == 0 && CollectionUtils.isNotEmpty(evalConfigList)) {
                    for(EvalConfig evalConfig : evalConfigList) {
                        status = encodeModelData(evalConfig);
                        updateEvalEncodeDataPath(status, encodeRefModel, evalConfig);
                        if(status != 0) {
                            break;
                        }
                    }
                }
            }

            clearUp(ModelStep.ENCODE);
        } catch (Exception e) {
            LOG.error("Fail to run encoding.", e);
            status = 1;
        }

        return status;
    }

    private void updateModel(String encodeRefModel) throws IOException {
        ModelConfig encodeModel = loadSubModelConfig(encodeRefModel);
        encodeModel.setModelSetName(encodeRefModel);

        int featureCnt = 1;
        for(int i = 0; i < this.treeModel.getTrees().size(); i++) {
            featureCnt = featureCnt * this.treeModel.getTrees().get(i).size();
        }

        List<String> categoricalVars = new ArrayList<String>();
        for(int i = 0; i < featureCnt; i++) {
            categoricalVars.add("tree_vars_" + i);
        }

        String catVarFileName = encodeModel.getDataSet().getCategoricalColumnNameFile();
        if(StringUtils.isBlank(catVarFileName)) {
            catVarFileName = "categorical.column.names";
            encodeModel.getDataSet().setCategoricalColumnNameFile(catVarFileName);
        }
        FileUtils.writeLines(new File(encodeRefModel + File.separator + catVarFileName), categoricalVars);

        saveModelConfig(encodeRefModel, encodeModel);
    }

    private void updateTrainEncodeDataPath(int status, String encodeRefModel) throws IOException {
        if(status == 0 && StringUtils.isNotBlank(encodeRefModel)) {
            String delimiter = Environment
                    .getProperty(Constants.SHIFU_OUTPUT_DATA_DELIMITER, Constants.DEFAULT_DELIMITER);
            String encodeDataPath = this.pathFinder.getEncodeDataPath(null);

            ModelConfig encodeModel = loadSubModelConfig(encodeRefModel);
            encodeModel.getDataSet().setDataPath(encodeDataPath);
            encodeModel.getDataSet().setDataDelimiter(delimiter);
            encodeModel.getDataSet().setHeaderPath(encodeDataPath + File.separator + ".pig_header");
            encodeModel.getDataSet().setHeaderDelimiter(delimiter);

            encodeModel.getDataSet().setFilterExpressions(""); // remove filter

            saveModelConfig(encodeRefModel, encodeModel);
        }
    }

    private void updateEvalEncodeDataPath(int status, String encodeRefModel, EvalConfig evalConfig) throws IOException {
        if(status == 0 && StringUtils.isNotBlank(encodeRefModel)) {
            String delimiter = Environment
                    .getProperty(Constants.SHIFU_OUTPUT_DATA_DELIMITER, Constants.DEFAULT_DELIMITER);
            String encodeDataPath = this.pathFinder.getEncodeDataPath(evalConfig);

            ModelConfig encodeModel = loadSubModelConfig(encodeRefModel);
            EvalConfig encodeEvalConfig = encodeModel.getEvalConfigByName(evalConfig.getName());
            if (encodeEvalConfig == null) { // new EvalSet, add it to encode model
                encodeEvalConfig = evalConfig.clone();
                encodeModel.getEvals().add(encodeEvalConfig);
            }

            encodeEvalConfig.getDataSet().setDataPath(encodeDataPath);
            encodeEvalConfig.getDataSet().setDataDelimiter(delimiter);
            encodeEvalConfig.getDataSet().setHeaderPath(encodeDataPath + File.separator + ".pig_header");
            encodeEvalConfig.getDataSet().setHeaderDelimiter(delimiter);

            encodeEvalConfig.getDataSet().setFilterExpressions(""); // remove filter

            saveModelConfig(encodeRefModel, encodeModel);
        }
    }

    @SuppressWarnings("deprecation")
    private int encodeModelData(EvalConfig evalConfig) throws IOException {
        int status = 0;

        RawSourceData.SourceType sourceType = this.modelConfig.getDataSet().getSource();
        // clean up output directories
        ShifuFileUtils.deleteFile(pathFinder.getEncodeDataPath(evalConfig), sourceType);

        // prepare special parameters and execute pig
        Map<String, String> paramsMap = new HashMap<String, String>();

        paramsMap.put(Constants.SOURCE_TYPE, sourceType.toString());
        paramsMap.put("pathRawData", (evalConfig == null) ?
                modelConfig.getDataSetRawPath() : evalConfig.getDataSet().getDataPath());
        paramsMap.put("pathEncodeData", pathFinder.getEncodeDataPath(evalConfig));
        paramsMap.put("delimiter", CommonUtils.escapePigString(modelConfig.getDataSetDelimiter()));
        paramsMap.put("evalSetName", (evalConfig == null ? TRAINING_DATA_SET : evalConfig.getName()));

        paramsMap.put(Constants.IS_COMPRESS, "true");

        try {
            String encodePigPath = pathFinder.getScriptPath("scripts/EncodeData.pig");
            ;
            PigExecutor.getExecutor().submitJob(modelConfig, encodePigPath, paramsMap);

            Iterator<JobStats> iter = PigStats.get().getJobGraph().iterator();

            while(iter.hasNext()) {
                JobStats jobStats = iter.next();
                if(jobStats.getHadoopCounters() != null
                        && jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER) != null) {
                    long totalValidCount = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                            .getCounter("TOTAL_VALID_COUNT");
                    // If no basic record counter, check next one
                    if(totalValidCount == 0L) {
                        continue;
                    }
                    long invalidTagCount = jobStats.getHadoopCounters().getGroup(Constants.SHIFU_GROUP_COUNTER)
                            .getCounter("INVALID_TAG");

                    LOG.info("Total valid records {} after filtering, invalid tag records {}.", totalValidCount,
                            invalidTagCount);

                    if(totalValidCount > 0L && invalidTagCount * 1d / totalValidCount >= 0.8d) {
                        LOG.error(
                                "Too many invalid tags, please check you configuration on positive tags and negative tags.");
                        status = 1;
                    }
                }
                // only one pig job with such counters, break
                break;
            }
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_RUNNING_PIG_JOB, e);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

        return status;
    }

    private String getEncodeDataSet() {
        return getStringParam(this.params, ENCODE_DATA_SET);
    }

    private String getEncodeRefModel() {
        return getStringParam(this.params, ENCODE_REF_MODEL);
    }

    public String getModelName(int i) {
        String alg = this.modelConfig.getTrain().getAlgorithm();
        return String.format("model%s.%s", i, alg.toLowerCase());
    }
}