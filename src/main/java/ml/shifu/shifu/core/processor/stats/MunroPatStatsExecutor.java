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
import ml.shifu.shifu.pig.PigExecutor;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by zhanhu on 7/1/16.
 */
public class MunroPatStatsExecutor extends MapReducerStatsWorker {

    private static Logger log = LoggerFactory.getLogger(MunroPatStatsExecutor.class);

    public MunroPatStatsExecutor(BasicModelProcessor processor, ModelConfig modelConfig,
            List<ColumnConfig> columnConfigList, boolean isUpdateStatsOnly) {
        super(processor, modelConfig, columnConfigList, isUpdateStatsOnly);
    }

    @Override
    protected void runStatsPig(Map<String, String> paramsMap) throws Exception {
        log.info("Run MunroPat to stats ... ");

        PigExecutor.getExecutor().submitJob(modelConfig, pathFinder.getScriptPath("scripts/Stats.pig"), paramsMap,
                modelConfig.getDataSet().getSource(), super.pathFinder);
    }

}
