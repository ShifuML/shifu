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
package ml.shifu.eval;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.common.Step;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.processor.EvalModelProcessor;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.JSONUtils;

import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Evaluation step which is to call {@link EvalModelProcessor}.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class EvalStep extends Step<Void> {

    private final static Logger LOG = LoggerFactory.getLogger(EvalStep.class);

    public EvalStep(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, Map<String, Object> otherConfigs) {
        super(ModelStep.TRAIN, modelConfig, columnConfigList, otherConfigs);
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.common.Step#process()
     */
    @Override
    public Void process() throws IOException {
        LOG.info("Saving ModelConfig, ColumnConfig and then upload to HDFS ...");
        JSONUtils.writeValue(new File(pathFinder.getModelConfigPath(SourceType.LOCAL)), modelConfig);
        JSONUtils.writeValue(new File(pathFinder.getColumnConfigPath(SourceType.LOCAL)), columnConfigList);

        if(StringUtils.isNotBlank(modelConfig.getDataSet().getEnhanceColumnFile())){
            enhanceEvalData();
        }
        EvalModelProcessor processor = new EvalModelProcessor(
                ml.shifu.shifu.core.processor.EvalModelProcessor.EvalStep.RUN, super.otherConfigs);
        try {
            processor.run();
        } catch (Exception e) {
            LOG.error("Error in training", e);
        }
        return null;
    }

    private void enhanceEvalData(){
        for(EvalConfig evalConfig: modelConfig.getEvals()){
            String enhancePigPath = pathFinder.getScriptPath("scripts/EnhanceColumn.pig");
            Map<String, String> paramsMap = new HashMap<String, String>();
            paramsMap.put(Constants.ENHANCE_DATA_TYPE, "train");
            paramsMap.put(Constants.EVAL_SET_NAME, evalConfig.getName());
            paramsMap.put(Constants.IS_COMPRESS, "false");
            try {
                PigExecutor.getExecutor().submitJob(modelConfig, enhancePigPath, paramsMap,
                        modelConfig.getDataSet().getSource(), super.pathFinder);
            } catch(IOException e) {
                LOG.error("Error happen on enhanced eval data set name " + evalConfig.getName());
            }
        }
    }

}
