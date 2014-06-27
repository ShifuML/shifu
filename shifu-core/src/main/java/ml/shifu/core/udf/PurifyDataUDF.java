/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.core.udf;

import ml.shifu.core.container.obj.EvalConfig;
import ml.shifu.core.core.DataPurifier;
import org.apache.pig.data.Tuple;

import java.io.IOException;


/**
 * PurifyDataUDF class purify the data for training and evaluation.
 * The setting for purify is in in @ModelConfig.dataSet.filterExpressions or
 *
 * @EvalConfig.dataSet.filterExpressions
 */
public class PurifyDataUDF extends AbstractTrainerUDF<Boolean> {

    private DataPurifier dataPurifier;

    /**
     * @param source
     * @param pathModelConfig
     * @param pathColumnConfig
     * @throws IOException
     */
    public PurifyDataUDF(String source, String pathModelConfig, String pathColumnConfig)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        dataPurifier = new DataPurifier(modelConfig);
    }

    /**
     * @param source
     * @param pathModelConfig
     * @param pathColumnConfig
     * @param evalSetName
     * @throws IOException
     */
    public PurifyDataUDF(String source, String pathModelConfig, String pathColumnConfig, String evalSetName)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        EvalConfig evalConfig = modelConfig.getEvalConfigByName(evalSetName);
        dataPurifier = new DataPurifier(evalConfig);
    }

    /* (non-Javadoc)
     * @see org.apache.pig.EvalFunc#exec(org.apache.pig.data.Tuple)
     */
    @Override
    public Boolean exec(Tuple input) throws IOException {
        return dataPurifier.isFilterOut(input);
    }

}
