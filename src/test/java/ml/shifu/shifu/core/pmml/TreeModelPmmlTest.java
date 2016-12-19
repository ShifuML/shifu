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

import java.util.List;
import java.util.Map;

import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ClassificationMap;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.TreeModelEvaluator;

public class TreeModelPmmlTest {

    @SuppressWarnings("unchecked")
    public void testTreeModelPMML() throws Exception {
        PMML pmml = PMMLUtils.loadPMML(getClass().getResource("/dttest/test/gbt.pmml").toString());
        TreeModelEvaluator evaluator = new TreeModelEvaluator(pmml);

        List<Map<FieldName, FieldValue>> input = CsvUtil.load(evaluator,
                getClass().getResource("/dttest/test/tmdata.csv").toString(), "|");

        for(Map<FieldName, FieldValue> maps: input) {
            switch(evaluator.getModel().getFunctionName()) {
                case REGRESSION:
                    Map<FieldName, Double> regressionTerm = (Map<FieldName, Double>) evaluator.evaluate(maps);
                    for(Map.Entry<FieldName, Double> entry: regressionTerm.entrySet())
                        System.out.println(entry.getValue() * 1000);
                    break;
                case CLASSIFICATION:
                    Map<FieldName, ClassificationMap<String>> classificationTerm = (Map<FieldName, ClassificationMap<String>>) evaluator
                            .evaluate(maps);
                    for(ClassificationMap<String> cMap: classificationTerm.values()) {
                        for(Map.Entry<String, Double> entry: cMap.entrySet())
                            System.out.println(entry.getValue() * 1000);
                    }
                    break;
            }
        }
    }

}
