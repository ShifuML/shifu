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

import ml.shifu.shifu.util.updater.ColumnConfigUpdater;
import org.apache.commons.lang.StringUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import ml.shifu.common.Step;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.core.validator.ModelInspector.ModelStep;
import ml.shifu.shifu.util.CommonUtils;

/**
 * Init Step in Shifu to call ColumnConfig initialization.
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
            String[] dataInFirstLine = CommonUtils.takeFirstLine(modelConfig.getDataSetRawPath(),
                    modelConfig.getDataSetDelimiter(), modelConfig.getDataSet().getSource());
            if(fields.length != dataInFirstLine.length) {
                throw new IllegalArgumentException(
                        "Header length and data length are not consistent, please check you header setting and data set setting.");
            }
        } else {
            fields = CommonUtils.takeFirstLine(modelConfig.getDataSetRawPath(), StringUtils.isBlank(modelConfig
                    .getHeaderDelimiter()) ? modelConfig.getDataSetDelimiter() : modelConfig.getHeaderDelimiter(),
                    modelConfig.getDataSet().getSource());
            if(StringUtils.join(fields, "").contains(modelConfig.getTargetColumnName())) {
                // if first line contains target column name, we guess it is csv format and first line is header.
                isSchemaProvided = true;

                // first line of data meaning second line in data files excluding first header line
                String[] dataInFirstLine = CommonUtils.takeFirstTwoLines(modelConfig.getDataSetRawPath(),
                        StringUtils.isBlank(modelConfig.getHeaderDelimiter()) ? modelConfig.getDataSetDelimiter()
                                : modelConfig.getHeaderDelimiter(), modelConfig.getDataSet().getSource())[1];

                if(dataInFirstLine != null && fields.length != dataInFirstLine.length) {
                    throw new IllegalArgumentException(
                            "Header length and data length are not consistent, please check you header setting and data set setting.");
                }
                LOG.warn("No header path is provided, we will try to read first line and detect schema.");
                LOG.warn("Schema in ColumnConfig.json are named as first line of data set path.");
            } else {
                isSchemaProvided = false;
                LOG.warn("No header path is provided, we will try to read first line and detect schema.");
                LOG.warn("Schema in ColumnConfig.json are named as  index 0, 1, 2, 3 ...");
                LOG.warn("Please make sure weight column and tag column are also taking index as name.");
            }
        }

        columnConfigList = new ArrayList<ColumnConfig>();
        for(int i = 0; i < fields.length; i++) {
            ColumnConfig config = new ColumnConfig();
            config.setColumnNum(i);
            if(isSchemaProvided) {
                config.setColumnName(CommonUtils.getRelativePigHeaderColumnName(fields[i]));
            } else {
                config.setColumnName(i + "");
            }
            columnConfigList.add(config);
        }

        ColumnConfigUpdater.updateColumnConfigFlags(modelConfig, columnConfigList, ModelStep.INIT);

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
