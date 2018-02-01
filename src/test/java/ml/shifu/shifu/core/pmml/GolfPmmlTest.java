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
package ml.shifu.shifu.core.pmml;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.NeuralNetworkEvaluator;
import org.junit.Test;

public class GolfPmmlTest {

    @SuppressWarnings({ "unchecked", "incomplete-switch" })
    @Test
    public void testOldNNPMMlModel() throws Exception {
        PMML pmml = PMMLUtils.loadPMML("src/test/resources/dttest/model/golf0.pmml");
        NeuralNetworkEvaluator evaluator = new NeuralNetworkEvaluator(pmml);
        List<Map<FieldName, FieldValue>> input = CsvUtil.load(evaluator, "src/test/resources/dttest/data/golf0.csv",
                "\\|");

        for(Map<FieldName, FieldValue> map: input) {
            Map<String, Object> newMap = new HashMap<String, Object>();
            Map<FieldName, Double> regressionTerm = (Map<FieldName, Double>) evaluator.evaluate(map);
            double pmmlScore = 0d;
            for(Map.Entry<FieldName, Double> entry: regressionTerm.entrySet()) {
                pmmlScore = entry.getValue();
            }
            for(Entry<FieldName, FieldValue> entry: map.entrySet()) {
                FieldName key = entry.getKey();
                FieldValue value = entry.getValue();

                switch(value.getOpType()) {
                    case CONTINUOUS:
                        newMap.put(key.getValue(), Double.parseDouble(value.getValue().toString()));
                        break;
                    case CATEGORICAL:
                        newMap.put(key.getValue(), value.getValue().toString());
                        break;
                }
            }
            System.out.println(pmmlScore);
        }
    }
    
    @SuppressWarnings("unchecked")
    @Test
    public void testNewPMMLModel() throws Exception {
        // model with segment expansion
        PMML pmml = PMMLUtils.loadPMML("src/test/resources/dttest/model/golf0-new.pmml");
        NeuralNetworkEvaluator evaluator = new NeuralNetworkEvaluator(pmml);
        List<Map<FieldName, FieldValue>> input = CsvUtil.load(evaluator, "src/test/resources/dttest/data/golf0-new.csv",
                "\\|");

        for(Map<FieldName, FieldValue> map: input) {
            Map<String, Object> newMap = new HashMap<String, Object>();
            Map<FieldName, Double> regressionTerm = (Map<FieldName, Double>) evaluator.evaluate(map);
            double pmmlScore = 0d;
            for(Map.Entry<FieldName, Double> entry: regressionTerm.entrySet()) {
                pmmlScore = entry.getValue();
            }
            for(Entry<FieldName, FieldValue> entry: map.entrySet()) {
                FieldName key = entry.getKey();
                FieldValue value = entry.getValue();

                switch(value.getOpType()) {
                    case CONTINUOUS:
                        newMap.put(key.getValue(), Double.parseDouble(value.getValue().toString()));
                        break;
                    case CATEGORICAL:
                        newMap.put(key.getValue(), value.getValue().toString());
                        break;
                }
            }
            System.out.println(pmmlScore);
        }
    }


}
