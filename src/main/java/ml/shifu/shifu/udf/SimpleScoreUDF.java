/**
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

import ml.shifu.shifu.container.CaseScoreResult;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ModelRunner;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.collections.MapUtils;
import org.apache.pig.data.Tuple;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.encog.ml.BasicML;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * SimpleScoreUDF class calculate the average score for evaluation data
 */
public class SimpleScoreUDF extends AbstractTrainerUDF<Integer> {

    private String targetColumnName;
    private List<String> negTags;
    private List<String> posTags;

    private String[] header;
    private ModelRunner modelRunner;

    public SimpleScoreUDF(String source, String pathModelConfig, String pathColumnConfig, String pathHeader,
            String delimiter) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

        SourceType sourceType = SourceType.valueOf(source);

        List<BasicML> models = CommonUtils.loadBasicModels(modelConfig, columnConfigList, null, sourceType);
        this.header = CommonUtils.getHeaders(pathHeader, delimiter, sourceType);
        modelRunner = new ModelRunner(modelConfig, columnConfigList, this.header, modelConfig.getDataSetDelimiter(),
                models);

        targetColumnName = columnConfigList.get(tagColumnNum).getColumnName();
        log.debug("Target Column Name: " + targetColumnName);

        negTags = modelConfig.getNegTags();
        posTags = modelConfig.getPosTags();
    }

    public Integer exec(Tuple input) throws IOException {
        CaseScoreResult cs = modelRunner.compute(input);
        if(cs == null) {
            log.error("Get null result.");
            return null;
        }

        Map<String, String> rawDataMap = CommonUtils.convertDataIntoMap(input, this.header);
        if(MapUtils.isEmpty(rawDataMap)) {
            return null;
        }

        String tag = rawDataMap.get(targetColumnName);
        if(!(negTags.contains(tag) || posTags.contains(tag))) {
            // invalid record
            log.error("Detected invalid record. Its tag is - " + tag);
            return null;
        }

        return cs.getAvgScore();
    }

    public Schema outputSchema(Schema input) {
        return null;
    }
}
