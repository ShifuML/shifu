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
import ml.shifu.guagua.util.MemoryUtils;
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

public class TreeModelEvalAndScoreTest {

    private TreeModel model;

    private IndependentTreeModel iTreeModel;

    @BeforeClass
    public void setUp() throws IOException {
        String modelPath = "src/test/resources/dttest/model/cam4.gbt";
        FileInputStream fi = null;
        try {
            fi = new FileInputStream(modelPath);
            // long start = System.nanoTime();
            iTreeModel = IndependentTreeModel.loadFromStream(fi, true);
            // System.out.println(iTreeModel.getTrees().get(0).size() + " " + SizeEstimator.estimate(iTreeModel) + " "
            // + TimeUnit.NANOSECONDS.toMillis(System.nanoTime() - start) + "ms");
            // System.out.println(SizeEstimator.estimate(iTreeModel));
            // System.out.println(SizeEstimator.estimate(iTreeModel.getTrees()));
            // System.out.println(SizeEstimator.estimate(iTreeModel.getNumNameMapping()));
            // System.out.println(SizeEstimator.estimate(iTreeModel.getCategoricalColumnNameNames()));
            // System.out.println(SizeEstimator.estimate(iTreeModel.getColumnNumIndexMapping()));

        } finally {
            fi.close();
        }

        System.gc();
        System.out.println(MemoryUtils.getRuntimeMemoryStats());

        fi = null;
        try {
            fi = new FileInputStream(modelPath);
            model = TreeModel.loadFromStream(fi, true);
        } finally {
            fi.close();
        }

        System.gc();
        // System.out.println(MemoryUtils.getRuntimeMemoryStats());
    }

    @Test
    public void testEvalScore() throws IOException {
        List<BasicML> models = new ArrayList<BasicML>();
        models.add(model);

        ModelConfig modelConfig = CommonUtils.loadModelConfig("src/test/resources/dttest/newconfig/ModelConfig.json",
                SourceType.LOCAL);
        List<ColumnConfig> columnConfigList = CommonUtils.loadColumnConfigList(
                "src/test/resources/dttest/newconfig/ColumnConfig.json", SourceType.LOCAL);

        Scorer scorer = new Scorer(models, columnConfigList, "GBT", modelConfig);

        List<String> lines = FileUtils.readLines(new File("src/test/resources/dttest/data/evaldata.csv"));

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
                map.put(headers[j], data[j]);
                mapObj.put(headers[j], data[j]);
            }
            @SuppressWarnings("unused")
            double[] scores = iTreeModel.compute(mapObj);

            @SuppressWarnings("unused")
            ScoreObject scoreObject = scorer.score(map);
            // System.out.println("Eval score is: " + scoreObject.getMeanScore() / 1000 + "; shifu score: " + scores[0]
            // + "; raw eval score is " + Double.parseDouble(map.get("model0").toString()) / 1000L);
        }

    }

}
