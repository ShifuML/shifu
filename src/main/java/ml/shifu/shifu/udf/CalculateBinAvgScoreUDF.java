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

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.util.CommonUtils;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;

import java.io.IOException;

/**
 * CalculateBinAvgScoreUDF class is to calculate the average score for each bin
 */
public class CalculateBinAvgScoreUDF extends AbstractTrainerUDF<Tuple> {

    public CalculateBinAvgScoreUDF(String source, String pathColumnConfig) throws Exception {
        super(source, pathColumnConfig);
    }

    public Tuple exec(Tuple input) throws IOException {
        TupleFactory tupleFactory = TupleFactory.getInstance();

        if (input == null || input.size() == 0) {
            return null;
        }

        Integer columnNum = (Integer) input.get(0);
        DataBag bag = (DataBag) input.get(1);

        ColumnConfig config = columnConfigList.get(columnNum);

        Double[] binScore = new Double[config.getBinLength()];
        Integer[] binCount = new Integer[config.getBinLength()];

        for (int i = 0; i < binScore.length; i++) {
            binScore[i] = 0.0;
            binCount[i] = 0;
        }

        for (Tuple t : bag) {
            if ( t.get(1) == null || StringUtils.isBlank(t.get(1).toString()) ) {
                continue;
            }
            
            int binNum = CommonUtils.getBinNum(config, t.get(1).toString());
            // int binNum = CommonUtils.getBinNum(config.getBinBoundary(), Double.valueOf(t.get(1).toString()));
            Object scoreStr = t.get(2);
            if (scoreStr == null) {
                log.error(t.get(1).toString() + " has null value!");
                continue;
            }

            binScore[binNum] += Double.valueOf(t.get(2).toString());
            binCount[binNum]++;
        }

        for (int i = 0; i < binScore.length; i++) {
            binScore[i] /= binCount[i];
        }

        Tuple tuple = tupleFactory.newTuple();
        tuple.append(columnNum);
        for (Double score : binScore) {
            tuple.append((int) Math.round(score));
        }
        return tuple;
    }

}
