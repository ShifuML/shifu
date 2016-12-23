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

import ml.shifu.shifu.core.DataSampler;
import ml.shifu.shifu.exception.ShifuErrorCode;
import ml.shifu.shifu.exception.ShifuException;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;

import java.io.IOException;
import java.util.List;

/**
 * DataFilterUDF class is used to filter data. This is used when sampling the data.
 */
public class DataFilterUDF extends AbstractTrainerUDF<Tuple> {

    private List<String> negTags;
    private List<String> posTags;

    private Double sampleRate;
    private Boolean sampleNegOnly;

    public DataFilterUDF(String source, String pathModelConfig, String pathColumnConfig, String sampleRate, String sampleNegOnly) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

        this.sampleRate = Double.valueOf(sampleRate);
        this.sampleNegOnly = Boolean.valueOf(sampleNegOnly);

        negTags = modelConfig.getNegTags();
        posTags = modelConfig.getPosTags();
    }

    public Tuple exec(Tuple input) throws IOException {
        Tuple tuple = TupleFactory.getInstance().newTuple();

        if (input.size() < this.columnConfigList.size()) {
            throw new ShifuException(ShifuErrorCode.ERROR_NO_EQUAL_COLCONFIG);
        }

        List<Object> filteredData = DataSampler.filter(tagColumnNum, posTags, negTags, input.getAll(), sampleRate, sampleNegOnly);

        if (filteredData == null) {
            return null;
        }

        for (Object o : filteredData) {
            tuple.append(o);
        }

        return tuple;
    }

    public Schema outputSchema(Schema input) {
        return null;
    }
}
