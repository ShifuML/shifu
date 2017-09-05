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

import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.core.DataPurifier;
import ml.shifu.shifu.util.Constants;

import org.apache.pig.data.Tuple;
import org.apache.pig.tools.pigstats.PigStatusReporter;

import java.io.IOException;

/**
 * PurifyDataUDF class purify the data for training and evaluation.
 * The setting for purify is in in @ModelConfig.dataSet.filterExpressions or
 */
public class PurifyDataUDF extends AbstractTrainerUDF<Boolean> {

    private DataPurifier dataPurifier;

    public PurifyDataUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        dataPurifier = new DataPurifier(modelConfig);
    }

    public PurifyDataUDF(String source, String pathModelConfig, String pathColumnConfig, String evalSetName)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        EvalConfig evalConfig = modelConfig.getEvalConfigByName(evalSetName);
        dataPurifier = new DataPurifier(evalConfig);
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
        Boolean filterOut = dataPurifier.isFilter(input);
        if(filterOut != null && !filterOut) {
            // update model run time for stats
            if(isPigEnabled(Constants.SHIFU_GROUP_COUNTER, "FILTER_OUT_COUNT")) {
                PigStatusReporter.getInstance().getCounter(Constants.SHIFU_GROUP_COUNTER, "FILTER_OUT_COUNT")
                        .increment(1);
            }
        }
        return filterOut;
    }

}
