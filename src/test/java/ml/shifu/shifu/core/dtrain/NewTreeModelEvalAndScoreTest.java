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
package ml.shifu.shifu.core.dtrain;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.guagua.util.FileUtils;
import ml.shifu.shifu.container.ScoreObject;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.Scorer;
import ml.shifu.shifu.core.TreeModel;
import ml.shifu.shifu.core.dtrain.dt.IndependentTreeModel;
import ml.shifu.shifu.util.CommonUtils;

import org.encog.ml.BasicML;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

public class NewTreeModelEvalAndScoreTest {

    private TreeModel model;

    private IndependentTreeModel iTreeModel;

    @BeforeClass
    public void setUp() throws IOException {
        String modelPath = "src/test/resources/dttest/model/newatom.gbt";
        FileInputStream fi = null;
        try {
            fi = new FileInputStream(modelPath);
            iTreeModel = IndependentTreeModel.loadFromStream(fi, true);
        } finally {
            fi.close();
        }

        fi = null;
        try {
            fi = new FileInputStream(modelPath);
            model = TreeModel.loadFromStream(fi, true);
        } finally {
            fi.close();
        }
    }

//    @Test
//    public void testNewModel() throws IOException {
//        ModelConfig modelConfig = CommonUtils.loadModelConfig(
//                "src/test/resources/dttest/newatomconfig/ModelConfig.json", SourceType.LOCAL);
//        List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList(
//                "src/test/resources/dttest/newatomconfig/ColumnConfig.json", SourceType.LOCAL);
//
//        FileOutputStream fileOutput = new FileOutputStream("atom22.gbt");
//
//        BinaryDTSerializer.save(modelConfig, columnConfigList, iTreeModel.getTrees(), "squared", 807, fileOutput);
//        
//        fileOutput.close();
//    }

    @Test
    public void testEvalScore() throws IOException {
        List<BasicML> models = new ArrayList<BasicML>();
        models.add(model);

        ModelConfig modelConfig = CommonUtils.loadModelConfig(
                "src/test/resources/dttest/newatomconfig/ModelConfig.json", SourceType.LOCAL);
        List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList(
                "src/test/resources/dttest/newatomconfig/ColumnConfig.json", SourceType.LOCAL);

        Scorer scorer = new Scorer(models, columnConfigList, "GBT", modelConfig);

        List<String> lines = FileUtils.readLines(new File("src/test/resources/dttest/data/newatom.csv"));

        if(lines.size() <= 1) {
            return;
        }
        String[] headers = CommonUtils.split(lines.get(0), "|");
        // score with format <String, String>
        for(int i = 1; i < lines.size(); i++) {
            Map<String, String> map = new HashMap<String, String>();
            Map<String, Object> mapObj = new HashMap<String, Object>();

            String[] data = CommonUtils.split(lines.get(i), "|");;
            // System.out.println("data len is " + data.length);
            if(data.length != headers.length) {
                System.out.println("One invalid input data");
                break;
            }
            for(int j = 0; j < headers.length; j++) {
                String header = headers[j];
                header = CommonUtils.getSimpleColumnName(header);
                map.put(header, data[j]);
                mapObj.put(header, data[j]);
            }

            // System.out.println(mapObj);
            double[] scores = iTreeModel.compute(mapObj);

            ScoreObject scoreObject = scorer.score(map);

            System.out.println("Eval score is: " + scoreObject.getMeanScore() / 1000 + "; shifu score: " + scores[0]);
            // + "; raw eval score is " + Double.parseDouble(map.get("model0").toString()) / 1000L);
        }

    }

}
