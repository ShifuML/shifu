/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.guagua.util.FileUtils;
import ml.shifu.shifu.core.dtrain.nn.IndependentNNModel;
import ml.shifu.shifu.util.CommonUtils;

public class IndependentNNModelTest {

    private IndependentNNModel nnModel;

//    @BeforeClass
    public void setUp() throws IOException {
        String modelPath = "src/test/resources/dttest/model/inde.nn";
        FileInputStream fi = null;
        try {
            fi = new FileInputStream(modelPath);
            nnModel = IndependentNNModel.loadFromStream(fi);
        } finally {
            fi.close();
        }
    }

//    @Test
    public void testEvalScore() throws IOException {
        List<String> headerList = FileUtils.readLines(new File(
                "src/test/resources/example/cancer-judgement/DataStore/EvalSet1/.pig_header"));
        String[] headers = CommonUtils.split(headerList.get(0), "|");

        List<String> lines = FileUtils.readLines(new File(
                "src/test/resources/example/cancer-judgement/DataStore/EvalSet1/part-00"));
        // score with format <String, String>
        for(int i = 0; i < lines.size(); i++) {
            Map<String, String> map = new HashMap<String, String>();
            Map<String, Object> mapObj = new HashMap<String, Object>();

            String[] data = CommonUtils.split(lines.get(i), "|");;
            if(data.length != headers.length) {
                System.out.println("One invalid input data");
                continue;
            }
            for(int j = 0; j < headers.length; j++) {
                map.put(headers[j], data[j]);
                mapObj.put(headers[j], data[j]);
            }
            double[] scores = nnModel.compute(mapObj);
            System.out.println("score is " + Arrays.toString(scores));
        }

    }

}
