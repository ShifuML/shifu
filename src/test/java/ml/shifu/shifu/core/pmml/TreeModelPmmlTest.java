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

import java.io.FileInputStream;
import java.io.InputStream;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.core.dtrain.dt.TreeNode;

import org.apache.commons.io.IOUtils;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.ClassificationMap;
import org.jpmml.evaluator.FieldValue;
import org.jpmml.evaluator.MiningModelEvaluator;

public class TreeModelPmmlTest {

    // @Test
    public void testTreeModel() throws Exception {
        InputStream is = null;
        try {
            is = new FileInputStream("src/test/resources/dttest/model/model-5.gbt");
            IndependentTreeModel model = IndependentTreeModel.loadFromStream(is);
            @SuppressWarnings("unused")
            List<TreeNode> trees = model.getTrees();
            // for(TreeNode treeNode: trees) {
            // System.out.println(treeNode.getNode().toTree());
            // }
            PMML pmml = PMMLUtils.loadPMML("src/test/resources/dttest/model/model-5.pmml");
            MiningModelEvaluator evaluator = new MiningModelEvaluator(pmml);
            List<Map<FieldName, FieldValue>> input = CsvUtil.load(evaluator,
                    "src/test/resources/dttest/data/tmdata-1.csv", "\\|");

            for(Map<FieldName, FieldValue> map: input) {
                Map<String, Object> newMap = new HashMap<String, Object>();
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
                double[] results = model.compute(newMap);
                System.out.println(newMap);
                System.out.println(results[0] * 1000);
            }
        } catch (Exception e) {
            e.printStackTrace();
            throw e;
        } finally {
            IOUtils.closeQuietly(is);
        }

    }

    @SuppressWarnings("unchecked")
    // @Test
    public void testTreeModelPMML() throws Exception {
        PMML pmml = PMMLUtils.loadPMML("src/test/resources/dttest/model/model-5.pmml");
        MiningModelEvaluator evaluator = new MiningModelEvaluator(pmml);

        System.out.println(evaluator.getActiveFields());

        List<Map<FieldName, FieldValue>> input = CsvUtil.load(evaluator, "src/test/resources/dttest/data/tmdata-1.csv",
                "\\|");

        for(Map<FieldName, FieldValue> maps: input) {
            switch(evaluator.getModel().getFunctionName()) {
                case REGRESSION:
                    System.out.println(maps);
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
