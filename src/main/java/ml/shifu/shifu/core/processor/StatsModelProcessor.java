/*
 * Copyright [2012-2014] PayPal Software Foundation
 * <p/>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p/>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p/>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.processor;

import ml.shifu.shifu.container.obj.ModelStatsConf;
import ml.shifu.shifu.core.processor.stats.*;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.util.CommonUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * statistics, max/min/avg/std for each column dataset if it's numerical
 */
public class StatsModelProcessor extends BasicModelProcessor implements Processor {

    private final static Logger log = LoggerFactory.getLogger(StatsModelProcessor.class);

    /**
     * runner for statistics
     */
    @Override
    public int run() throws Exception {
        log.info("Step Start: stats");
        long start = System.currentTimeMillis();
        try {
            setUp(ModelStep.STATS);

            // User may change variable type after `shifu init`
            CommonUtils.updateColumnConfigFlags(this.modelConfig, this.columnConfigList);
            // after read forceSelet/forceRemove/categorical refresh local ColumnConfig.json
            saveColumnConfigListAndColumnStats(false);

            // resync ModelConfig.json/ColumnConfig.json to HDFS
            syncDataToHdfs(modelConfig.getDataSet().getSource());

            AbstractStatsExecutor statsExecutor = null;

            if (modelConfig.isMapReduceRunMode()) {
                if (modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.DynamicBinning)) {
                    statsExecutor = new DIBStatsExecutor(this, modelConfig, columnConfigList);
                } else if (modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.MunroPat)) {
                    statsExecutor = new MunroPatStatsExecutor(this, modelConfig, columnConfigList);
                } else if (modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.MunroPatI)) {
                    statsExecutor = new MunroPatIStatsExecutor(this, modelConfig, columnConfigList);
                } else if (modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.SPDT)) {
                    statsExecutor = new SPDTStatsExecutor(this, modelConfig, columnConfigList);
                } else if (modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.SPDTI)) {
                    statsExecutor = new SPDTIStatsExecutor(this, modelConfig, columnConfigList);
                } else {
                    statsExecutor = new SPDTIStatsExecutor(this, modelConfig, columnConfigList);
                }
            } else if (modelConfig.isLocalRunMode()) {
                statsExecutor = new AkkaStatsWorker(this, modelConfig, columnConfigList);
            } else {
                throw new ShifuException(ShifuErrorCode.ERROR_UNSUPPORT_MODE);
            }

            statsExecutor.doStats();

            syncDataToHdfs(modelConfig.getDataSet().getSource());

            clearUp(ModelStep.STATS);
        } catch (Exception e) {
            log.error("Error:", e);
            return -1;
        }

        log.info("Step Finished: stats with {} ms", (System.currentTimeMillis() - start));
        return 0;
    }

}
