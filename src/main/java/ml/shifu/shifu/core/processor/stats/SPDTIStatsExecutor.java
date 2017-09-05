/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.processor.stats;

import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.processor.BasicModelProcessor;
import ml.shifu.shifu.fs.ShifuFileUtils;
import ml.shifu.shifu.pig.PigExecutor;
import ml.shifu.shifu.util.Environment;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by zhanhu on 7/1/16.
 */
public class SPDTIStatsExecutor extends MapReducerStatsWorker {

    private static Logger log = LoggerFactory.getLogger(SPDTIStatsExecutor.class);

    public SPDTIStatsExecutor(BasicModelProcessor processor, ModelConfig modelConfig,
            List<ColumnConfig> columnConfigList) {
        super(processor, modelConfig, columnConfigList);
    }

    @Override
    protected void runStatsPig(Map<String, String> paramsMap) throws Exception {
        log.info("Run SPDTI to stats ... ");

        paramsMap.put("group_binning_parallel", Integer.toString(columnConfigList.size() / (5 * 8)));
        
        String expressionsAsString = super.modelConfig.getSegmentFilterExpressionsAsString();
        Environment.getProperties().put("shifu.segment.expressions", expressionsAsString);
        
        ShifuFileUtils.deleteFile(pathFinder.getUpdatedBinningInfoPath(modelConfig.getDataSet().getSource()),
                modelConfig.getDataSet().getSource());

        PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getScriptPath("scripts/StatsSpdtI.pig"),
                paramsMap, modelConfig.getDataSet().getSource(), super.pathFinder);

        // update
        log.info("Updating binning info ...");
        updateBinningInfoWithMRJob();
    }
}
