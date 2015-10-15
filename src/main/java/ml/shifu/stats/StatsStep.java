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
package ml.shifu.stats;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import ml.shifu.common.Step;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelStatsConf;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.processor.BasicModelProcessor;
import ml.shifu.shifu.core.processor.stats.AbstractStatsExecutor;
import ml.shifu.shifu.core.processor.stats.AkkaStatsWorker;
import ml.shifu.shifu.core.processor.stats.DIBStatsExecutor;
import ml.shifu.shifu.core.processor.stats.MunroPatIStatsExecutor;
import ml.shifu.shifu.core.processor.stats.MunroPatStatsExecutor;
import ml.shifu.shifu.core.processor.stats.SPDTIStatsExecutor;
import ml.shifu.shifu.core.processor.stats.SPDTStatsExecutor;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.JSONUtils;

import ml.shifu.shifu.util.updater.ColumnConfigUpdater;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Stats step for stats functions in SHiuf pipeline.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class StatsStep extends Step<List<ColumnConfig>> {

    private final static Logger LOG = LoggerFactory.getLogger(StatsStep.class);

    public StatsStep(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, Map<String, Object> otherConfigs) {
        super(ModelStep.STATS, modelConfig, columnConfigList, otherConfigs);
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
        try {

            // User may change variable type after `shifu init`
            ColumnConfigUpdater.updateColumnConfigFlags(this.modelConfig, this.columnConfigList, ModelStep.STATS);

            LOG.info("Saving ModelConfig, ColumnConfig and then upload to HDFS ...");
            JSONUtils.writeValue(new File(pathFinder.getModelConfigPath(SourceType.LOCAL)), modelConfig);
            JSONUtils.writeValue(new File(pathFinder.getColumnConfigPath(SourceType.LOCAL)), columnConfigList);

            if(SourceType.HDFS.equals(modelConfig.getDataSet().getSource())) {
                CommonUtils.copyConfFromLocalToHDFS(modelConfig, this.pathFinder);
            }

            AbstractStatsExecutor statsExecutor = null;

            if(modelConfig.isMapReduceRunMode()) {
                if(modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.DynamicBinning)) {
                    statsExecutor = new DIBStatsExecutor(new BasicModelProcessor(super.modelConfig,
                            super.columnConfigList, super.otherConfigs), modelConfig, columnConfigList);
                } else if(modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.MunroPat)) {
                    statsExecutor = new MunroPatStatsExecutor(new BasicModelProcessor(super.modelConfig,
                            super.columnConfigList, super.otherConfigs), modelConfig, columnConfigList);
                } else if(modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.MunroPatI)) {
                    statsExecutor = new MunroPatIStatsExecutor(new BasicModelProcessor(super.modelConfig,
                            super.columnConfigList, super.otherConfigs), modelConfig, columnConfigList);
                } else if(modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.SPDT)) {
                    statsExecutor = new SPDTStatsExecutor(new BasicModelProcessor(super.modelConfig,
                            super.columnConfigList, super.otherConfigs), modelConfig, columnConfigList);
                } else if(modelConfig.getBinningAlgorithm().equals(ModelStatsConf.BinningAlgorithm.SPDTI)) {
                    statsExecutor = new SPDTIStatsExecutor(new BasicModelProcessor(super.modelConfig,
                            super.columnConfigList, super.otherConfigs), modelConfig, columnConfigList);
                } else {
                    statsExecutor = new SPDTIStatsExecutor(new BasicModelProcessor(super.modelConfig,
                            super.columnConfigList, super.otherConfigs), modelConfig, columnConfigList);
                }
            } else if(modelConfig.isLocalRunMode()) {
                statsExecutor = new AkkaStatsWorker(new BasicModelProcessor(super.modelConfig, super.columnConfigList,
                        super.otherConfigs), modelConfig, columnConfigList);
            } else {
                throw new ShifuException(ShifuErrorCode.ERROR_UNSUPPORT_MODE);
            }

            statsExecutor.doStats();

            if(SourceType.HDFS.equals(modelConfig.getDataSet().getSource())) {
                CommonUtils.copyConfFromLocalToHDFS(modelConfig, this.pathFinder);
            }
        } catch (Exception e) {
            LOG.error("Error:", e);
        }

        LOG.info("Step Finished: stats with {} ms", (System.currentTimeMillis() - start));
        return columnConfigList;
    }

}
