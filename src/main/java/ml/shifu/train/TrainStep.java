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
package ml.shifu.train;

import java.io.File;
import java.io.IOException;
import java.util.List;
import java.util.Map;

import ml.shifu.common.Step;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.processor.TrainModelProcessor;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.util.JSONUtils;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Train step which is to call Shifu training which is the same as {@link TrainModelProcessor}.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class TrainStep extends Step<Void> {

    private final static Logger LOG = LoggerFactory.getLogger(TrainStep.class);

    public TrainStep(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, Map<String, Object> otherConfigs) {
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

        TrainModelProcessor processor = new TrainModelProcessor(super.otherConfigs);
        try {
            processor.run();
        } catch (Exception e) {
            LOG.error("Error in training", e);
        }
        return null;
    }

}
