package ml.shifu.shifu.core.pmml;

import ml.shifu.shifu.combo.CsvFile;
import ml.shifu.shifu.core.pmml.builder.creator.AbstractSpecifCreator;
import org.dmg.pmml.DataType;
import org.dmg.pmml.FieldName;
import org.dmg.pmml.PMML;
import org.jpmml.evaluator.*;
import org.jpmml.evaluator.neural_network.NeuralNetworkEvaluator;
import org.testng.Assert;

import java.util.*;

/**
 * Copyright [2013-2018] PayPal Software Foundation
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License")
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 **/

public class PmmlSpecValidationTest {

    public static final double EPS = 1e-6;

    // @Test
    public void testAngModel() throws Exception {
        String pmmlSpecPath = "final_varsel_bag0_1.pmml";
        String testDataPath = "sample.csv";
        String delimiter = "\\|";
        String scoreFields = "model_score";
        
        doValidation(pmmlSpecPath, testDataPath, delimiter, scoreFields);
    }

    // @Test
    public void testAngModel2() throws Exception {
        String pmmlSpecPath = "final_varsel_bag0_1.pmml";
        String testDataPath = "sample_20k.csv";
        String delimiter = "\u0007";
        String scoreFields = "mean";

        boolean result = doValidation(pmmlSpecPath, testDataPath, delimiter, scoreFields);
        Assert.assertTrue(result);
    }

     @SuppressWarnings("unchecked")
    private boolean doValidation(String pmmlPath, String DataPath, String delimiter, String scoreName) throws Exception {
        PMML pmml = PMMLUtils.loadPMML(pmmlPath);
        NeuralNetworkEvaluator evaluator = new NeuralNetworkEvaluator(pmml);


        List<TargetField> targetFields = evaluator.getTargetFields();
        CsvFile evalData = new CsvFile(DataPath, delimiter, true);
        Iterator<Map<String, String>> iterator = evalData.iterator();

        int mismatchCnt = 0;

        while ( iterator.hasNext() ) {
            Map<String, String> rawInput = iterator.next();
            Map<FieldName, FieldValue> maps = convertRawIntoInput(evaluator, rawInput);

            double pmmlScore = 0.0;

            switch (evaluator.getModel().getMiningFunction()) {
                case REGRESSION:
                    if ( targetFields.size() == 1 ) {
                        Map<FieldName, Double> regressionTerm = (Map<FieldName, Double>) evaluator.evaluate(maps);
                        pmmlScore = regressionTerm.get(new FieldName(AbstractSpecifCreator.FINAL_RESULT));
                    } else {
                        Map<FieldName, Double> regressionTerm = (Map<FieldName, Double>) evaluator.evaluate(maps);
                        List<FieldName> outputFieldList = new ArrayList<FieldName>(regressionTerm.keySet());
                        Collections.sort(outputFieldList, new Comparator<FieldName>() {
                            @Override
                            public int compare(FieldName a, FieldName b) {
                                return a.getValue().compareTo(b.getValue());
                            }
                        });
                        for (int i = 0; i < outputFieldList.size(); i ++ ) {
                            FieldName fieldName = outputFieldList.get(i);
                            if ( fieldName.getValue().startsWith(AbstractSpecifCreator.FINAL_RESULT) ) {
                                pmmlScore = regressionTerm.get(fieldName);
                            }
                        }
                    }
                    break;
                case CLASSIFICATION:
                    Map<FieldName, Classification<Double>> classificationTerm = (Map<FieldName, Classification<Double>>) evaluator.evaluate(maps);
                    for (Classification<Double> cMap : classificationTerm.values())
                        for (Map.Entry<String,Value<Double>> entry : cMap.getValues().entrySet())
                            System.out.println(entry.getValue().getValue() * 1000);
                    break;
                default:
                    break;
            }

            double expectScore = Double.parseDouble(rawInput.get(scoreName));

            if ( Math.abs(expectScore - pmmlScore) > EPS ) {
                System.out.println(rawInput.get("trans_id") + "|" + expectScore + "|" + pmmlScore);
                mismatchCnt ++;
            }
        }

        return mismatchCnt == 0;
    }


    private Map<FieldName, FieldValue> convertRawIntoInput(NeuralNetworkEvaluator evaluator, Map<String, String> rawInput) {
        Map<FieldName, FieldValue> arguments = new HashMap<FieldName, FieldValue>();
        for(InputField inputField : evaluator.getInputFields()) {
            FieldName name = inputField.getName();
            if(rawInput.containsKey(name.getValue())) {
                String valStr = rawInput.get(name.getValue());
                if ( valStr.equalsIgnoreCase("NULL") ) {
                    valStr = "";
                }

                if( inputField.getDataType().equals(DataType.DOUBLE) ) {
                    Double value = null;
                    try {
                        value = Double.valueOf(valStr);
                    } catch (Exception e) {
                        value = null;
                    }
                    arguments.put(inputField.getName(), CsvUtil.prepare(inputField, value));
                } else {
                    arguments.put(inputField.getName(), CsvUtil.prepare(inputField, valStr));
                }
            } else {
                arguments.put(inputField.getName(), CsvUtil.prepare(inputField, null));
            }
        }

        return arguments;
    }

}
