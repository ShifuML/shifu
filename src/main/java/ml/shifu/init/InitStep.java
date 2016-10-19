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
package ml.shifu.init;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.common.Step;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.util.CommonUtils;

/**
 * TODO
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public class InitStep extends Step<List<ColumnConfig>> {

    private final static Logger LOG = LoggerFactory.getLogger(InitStep.class);

    public InitStep(ModelConfig modelConfig, List<ColumnConfig> columnConfigList, Map<String, Object> otherConfigs) {
        super(ModelStep.INIT, modelConfig, columnConfigList, otherConfigs);
    }

    /*
     * (non-Javadoc)
     * 
     * @see ml.shifu.common.Step#process()
     */
    @Override
    public List<ColumnConfig> process() throws IOException {
        // 1. init from header files
        initColumnConfigList();
        return this.columnConfigList;
    }

    private int initColumnConfigList() throws IOException {
        String[] fields = null;
        boolean isSchemaProvided = true;
        if(StringUtils.isNotBlank(modelConfig.getHeaderPath())) {
            fields = CommonUtils.getHeaders(modelConfig.getHeaderPath(), modelConfig.getHeaderDelimiter(), modelConfig
                    .getDataSet().getSource());
        } else {
            LOG.warn("No header path is provided, we will try to read first line and detect schema.");
            LOG.warn("Schema in ColumnConfig.json are named as  index 0, 1, 2, 3 ...");
            LOG.warn("Please make sure weight column and tag column are also taking index as name.");
            fields = CommonUtils.takeFirstLine(modelConfig.getDataSetRawPath(), modelConfig.getHeaderDelimiter(),
                    modelConfig.getDataSet().getSource());
            isSchemaProvided = false;
        }

        columnConfigList = new ArrayList<ColumnConfig>();
        for(int i = 0; i < fields.length; i++) {
            ColumnConfig config = new ColumnConfig();
            config.setColumnNum(i);
            if(isSchemaProvided) {
                config.setColumnName(fields[i]);
            } else {
                config.setColumnName(i + "");
            }
            columnConfigList.add(config);
        }

        CommonUtils.updateColumnConfigFlags(modelConfig, columnConfigList);

        boolean hasTarget = false;
        for(ColumnConfig config: columnConfigList) {
            if(config.isTarget()) {
                hasTarget = true;
            }
        }

        if(!hasTarget) {
            LOG.error("Target is not valid: " + modelConfig.getTargetColumnName());
            LOG.error("Please check your header file {} and your header delimiter {}", modelConfig.getHeaderPath(),
                    modelConfig.getHeaderDelimiter());
            return 1;
        }

        return 0;
    }

}
