/*
 * Copyright [2013-2016] PayPal Software Foundation
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
package ml.shifu.norm;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.common.Step;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.processor.NormalizeModelProcessor;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.JSONUtils;

import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Norm step includes some normalization functions and almost the same as {@link NormalizeModelProcessor}.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class NormStep extends Step<List<ColumnConfig>> {

    private final static Logger LOG = LoggerFactory.getLogger(NormStep.class);

    public NormStep(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, Map<String, Object> otherConfigs) {
        super(ModelStep.TRAIN, modelConfig, columnConfigList, otherConfigs);
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.common.Step#process()
     */
    @Override
    public List<ColumnConfig> process() throws IOException {
        LOG.info("Step Start: stats");
        long start = System.currentTimeMillis();

        LOG.info("Saving ModelConfig, ColumnConfig and then upload to HDFS ...");
        JSONUtils.writeValue(new File(pathFinder.getModelConfigPath(SourceType.LOCAL)), modelConfig);
        JSONUtils.writeValue(new File(pathFinder.getColumnConfigPath(SourceType.LOCAL)), columnConfigList);

        if(SourceType.HDFS.equals(modelConfig.getDataSet().getSource())) {
            CommonUtils.copyConfFromLocalToHDFS(modelConfig, this.pathFinder);
        }

        SourceType sourceType = modelConfig.getDataSet().getSource();

        ShifuFileUtils.deleteFile(pathFinder.getNormalizedDataPath(), sourceType);
        ShifuFileUtils.deleteFile(pathFinder.getNormalizedValidationDataPath(), sourceType);
        ShifuFileUtils.deleteFile(pathFinder.getSelectedRawDataPath(), sourceType);

        Map<String, String> paramsMap = new HashMap<String, String>();
        paramsMap.put("sampleRate", modelConfig.getNormalizeSampleRate().toString());
        paramsMap.put("sampleNegOnly", ((Boolean) modelConfig.isNormalizeSampleNegOnly()).toString());
        paramsMap.put("delimiter", CommonUtils.escapePigString(modelConfig.getDataSetDelimiter()));

        try {
            String normPigPath = null;
            if(modelConfig.getNormalize().getIsParquet()) {
                if(modelConfig.getBasic().getPostTrainOn()) {
                    normPigPath = pathFinder.getScriptPath("scripts/NormalizeWithParquetAndPostTrain.pig");
                } else {
                    LOG.info("Post train is disabled by 'postTrainOn=false'.");
                    normPigPath = pathFinder.getScriptPath("scripts/NormalizeWithParquet.pig");
                }
            } else {
                if(modelConfig.getBasic().getPostTrainOn()) {
                    // this condition is for comment, no matter post train enabled or not, only norm results will be
                    // stored since new post train solution
                }
                normPigPath = pathFinder.getScriptPath("scripts/Normalize.pig");
            }
            paramsMap.put(Constants.IS_COMPRESS, "true");
            paramsMap.put(Constants.IS_NORM_FOR_CLEAN, "false");
            PigExecutor.getExecutor().submitJob(modelConfig, normPigPath, paramsMap,
                    modelConfig.getDataSet().getSource(), super.pathFinder);
            if(StringUtils.isNotBlank(modelConfig.getValidationDataSetRawPath())) {
                paramsMap.put(Constants.IS_COMPRESS, "false");
                paramsMap.put(Constants.PATH_RAW_DATA, modelConfig.getValidationDataSetRawPath());
                paramsMap.put(Constants.PATH_NORMALIZED_DATA, pathFinder.getNormalizedValidationDataPath());
                PigExecutor.getExecutor().submitJob(modelConfig, normPigPath, paramsMap,
                        modelConfig.getDataSet().getSource(), super.pathFinder);
            }
        } catch (IOException e) {
            throw new ShifuException(ShifuErrorCode.ERROR_RUNNING_PIG_JOB, e);
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }

        LOG.info("Step Finished: stats with {} ms", (System.currentTimeMillis() - start));
        return columnConfigList;
    }

}
