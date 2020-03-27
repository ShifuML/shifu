/*
 * Copyright [2012-2014] PayPal Software Foundation
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
package ml.shifu.shifu.udf;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang3.StringUtils;
import org.apache.pig.data.Tuple;
import org.apache.pig.impl.util.UDFContext;
import org.apache.pig.tools.pigstats.PigStatusReporter;

import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.core.DataPurifier;
import ml.shifu.shifu.core.dtrain.CommonConstants;
import ml.shifu.shifu.util.CommonUtils;
import ml.shifu.shifu.util.Constants;

/**
 * PurifyDataUDF class purify the data for training and evaluation.
 * The setting for purify is in in @ModelConfig.dataSet.filterExpressions or
 */
public class PurifyDataUDF extends AbstractTrainerUDF<Boolean> {

    private DataPurifier dataPurifier;

    private List<DataPurifier> mtlDataPurifiers;

    private boolean isForNormMTLFilters = false;

    public PurifyDataUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        this(source, pathModelConfig, pathColumnConfig, "", "false");
    }

    public PurifyDataUDF(String source, String pathModelConfig, String pathColumnConfig, String evalSetName,
            String isForNormMTL) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

        boolean isForValidationDataSet = false;
        if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            isForValidationDataSet = Boolean.TRUE.toString().equalsIgnoreCase(UDFContext.getUDFContext().getJobConf()
                    .get(Constants.IS_VALIDATION_DATASET, Boolean.FALSE.toString()));
        }

        dataPurifier = new DataPurifier(modelConfig, columnConfigList, isForValidationDataSet);
        isForNormMTLFilters = Boolean.TRUE.toString().equalsIgnoreCase(isForNormMTL);
        if(isForNormMTLFilters && modelConfig.isMultiTask()) {
            String[] filters = CommonUtils.split(modelConfig.getDataSet().getFilterExpressions(),
                    CommonConstants.MTL_DELIMITER);
            if(filters != null && filters.length > 1) {
                if(filters.length != modelConfig.getMultiTaskTargetColumnNames().size()) {
                    throw new IllegalArgumentException(
                            "Size of multiple filter are not equals to size of mutiple target columns, please check ModelConfig#dataSet#targetColumnName and ModelConfig#dataSet#filterExpressions.");
                }
                mtlDataPurifiers = new ArrayList<>(filters.length);
                for(String filter: filters) {
                    mtlDataPurifiers.add(new DataPurifier(modelConfig, columnConfigList, filter));
                }
            }
        }
    }

    public PurifyDataUDF(String source, String pathModelConfig, String pathColumnConfig, String evalSetName)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

        boolean isForValidationDataSet = false;
        if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            isForValidationDataSet = Boolean.TRUE.toString().equalsIgnoreCase(UDFContext.getUDFContext().getJobConf()
                    .get(Constants.IS_VALIDATION_DATASET, Boolean.FALSE.toString()));
        }

        EvalConfig evalConfig = modelConfig.getEvalConfigByName(evalSetName);
        if(evalConfig == null) {
            dataPurifier = new DataPurifier(modelConfig, this.columnConfigList, isForValidationDataSet);
        } else {
            if(modelConfig.isMultiTask()) {
                String filterExpression = evalConfig.getDataSet().getFilterExpressions();
                if(StringUtils.isBlank(filterExpression)) {
                    dataPurifier = new DataPurifier(this.columnConfigList, evalConfig);
                } else {
                    String[] filters = CommonUtils.split(filterExpression, CommonConstants.MTL_DELIMITER);
                    if(filters != null && filters.length > 1) {
                        dataPurifier = new DataPurifier(modelConfig, columnConfigList,
                                filters[modelConfig.getMtlIndex()]);
                    } else {
                        dataPurifier = new DataPurifier(this.columnConfigList, evalConfig);
                    }
                }
            } else {
                dataPurifier = new DataPurifier(this.columnConfigList, evalConfig);
            }
        }
    }

    /*
     * (non-Javadoc)
     * 
     * @see org.apache.pig.EvalFunc#exec(org.apache.pig.data.Tuple)
     */
    @SuppressWarnings("deprecation")
    @Override
    public Boolean exec(Tuple input) throws IOException {
        // update model run time for stats
        if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "TOTAL_VALID_COUNT")) {
            PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "TOTAL_VALID_COUNT").increment(1);
        }
        if(CollectionUtils.isEmpty(this.mtlDataPurifiers)) {
            Boolean filterOut = dataPurifier.isFilter(input);
            if(filterOut != null && !filterOut) {
                // update model run time for stats
                if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "FILTER_OUT_COUNT")) {
                    PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "FILTER_OUT_COUNT")
                            .increment(1);
                }
            }
            return filterOut;
        } else {
            boolean result = false;
            for(DataPurifier dp: mtlDataPurifiers) {
                Boolean fi = dp.isFilter(input);
                result = result || (fi != null && fi.booleanValue());
                // if one is true, return true;
            }
            if(!result) {
                if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "FILTER_OUT_COUNT")) {
                    PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "FILTER_OUT_COUNT")
                            .increment(1);
                }
            }
            return result;
        }
    }

}
