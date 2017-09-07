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

import ml.shifu.shifu.column.NSColumn;
import ml.shifu.shifu.container.CaseScoreResult;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.ModelRunner;
import ml.shifu.shifu.util.CommonUtils;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.encog.ml.BasicML;

import java.io.IOException;
import java.util.List;
import java.util.Map;

/**
 * FullScoreUDF class it to calculate the full score of evaluation data
 * Full score contains avg/max/min/model0/...
 */
public class FullScoreUDF extends AbstractTrainerUDF<Tuple> {

    private String[] header;
    private ModelRunner modelRunner;

    public FullScoreUDF(String source, String pathModelConfig, String pathColumnConfig, String pathHeader,
            String delimiter) throws Exception {
        super(source, pathModelConfig, pathColumnConfig);

        List<BasicML> models = CommonUtils.loadBasicModels(modelConfig, null, SourceType.valueOf(source));
        this.header = CommonUtils.getHeaders(pathHeader, delimiter, SourceType.valueOf(source));
        modelRunner = new ModelRunner(modelConfig, columnConfigList, this.header, modelConfig.getDataSetDelimiter(),
                models);
    }

    public Tuple exec(Tuple input) throws IOException {
        Map<NSColumn, String> rawDataNsMap = CommonUtils.convertDataIntoNsMap(input, this.header, 0);

        CaseScoreResult cs = modelRunner.computeNsData(rawDataNsMap);
        if(cs == null) {
            log.error("Get null result.");
            return null;
        }

        Tuple tuple = TupleFactory.getInstance().newTuple();

        tuple.append(cs.getAvgScore());
        tuple.append(cs.getMaxScore());
        tuple.append(cs.getMinScore());

        for(double score: cs.getScores()) {
            tuple.append(score);
        }

        List<String> metaList = modelConfig.getMetaColumnNames();
        for(String meta: metaList) {
            tuple.append(rawDataNsMap.get(new NSColumn(meta)));
        }

        return tuple;
    }

    public Schema outputSchema(Schema input) {
        return null;
    }
}
