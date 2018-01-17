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
package ml.shifu.shifu.core;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import ml.shifu.shifu.container.ScoreObject;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnType;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.core.alg.SVMTrainer;
import ml.shifu.shifu.util.Constants;

import org.apache.commons.io.FileUtils;
import org.encog.ml.BasicML;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;

public class ScorerTest {

    private ModelConfig modelConfig;
    List<BasicML> models = new ArrayList<BasicML>();
    MLDataSet set = new BasicMLDataSet();

    @BeforeClass
    public void setup() throws IOException {
        modelConfig = ModelConfig.createInitModelConfig(".", ALGORITHM.NN, ".", false);

        modelConfig.getTrain().getParams().put("Propagation", "B");
        modelConfig.getTrain().getParams().put("NumHiddenLayers", 2);
        modelConfig.getTrain().getParams().put("LearningRate", 0.5);
        List<Integer> nodes = new ArrayList<Integer>();
        nodes.add(3);
        nodes.add(4);
        List<String> func = new ArrayList<String>();
        func.add("linear");
        func.add("tanh");
        modelConfig.getTrain().getParams().put("NumHiddenNodes", nodes);
        modelConfig.getTrain().getParams().put("ActivationFunc", func);

        NNTrainer trainer = new NNTrainer(modelConfig, 0, false);

        double[] input = { 0., 0., };
        double[] ideal = { 1. };
        MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));
        set.add(pair);

        input = new double[] { 0., 1., };
        ideal = new double[] { 0. };
        pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));
        set.add(pair);

        input = new double[] { 1., 0., };
        ideal = new double[] { 0. };
        pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));
        set.add(pair);

        input = new double[] { 1., 1., };
        ideal = new double[] { 1. };
        pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));
        set.add(pair);

        trainer.setTrainSet(set);
        trainer.setValidSet(set);

        trainer.train();

        modelConfig.getTrain().setAlgorithm("SVM");
        modelConfig.getTrain().getParams().put("Kernel", "rbf");
        modelConfig.getTrain().getParams().put("Const", 0.1);
        modelConfig.getTrain().getParams().put("Gamma", 1.0);
        modelConfig.getVarSelect().setFilterNum(2);

        SVMTrainer svm = new SVMTrainer(modelConfig, 1, false);
        svm.setTrainSet(set);
        svm.setValidSet(set);

        svm.train();

        models.add(trainer.getNetwork());
        models.add(svm.getSVM());

    }

//    @Test
    public void scoreTest() {
        List<ColumnConfig> list = new ArrayList<ColumnConfig>();
        ColumnConfig col = new ColumnConfig();
        col.setColumnType(ColumnType.N);
        col.setColumnName("A");
        col.setColumnNum(0);
        col.setFinalSelect(true);
        list.add(col);

        col = new ColumnConfig();
        col.setColumnType(ColumnType.N);
        col.setColumnName("B");
        col.setColumnNum(1);
        col.setFinalSelect(true);
        list.add(col);

        Scorer s = new Scorer(models, list, "NN", modelConfig);

        double[] input = { 0., 0., };
        double[] ideal = { 1. };
        MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));

        ScoreObject o = s.score(pair, null);
        List<Double> scores = o.getScores();

        Assert.assertTrue(scores.get(0) > 400);
        Assert.assertTrue(scores.get(1) == 1000);
    }

//    @Test
    public void scoreNull() {
        Scorer s = new Scorer(models, null, "NN", modelConfig);

        Assert.assertNull(s.score(null, null));
    }

//    @Test
    public void scoreModelsException() {
        List<ColumnConfig> list = new ArrayList<ColumnConfig>();
        ColumnConfig col = new ColumnConfig();
        col.setColumnType(ColumnType.N);
        col.setColumnName("A");
        col.setColumnNum(0);
        col.setFinalSelect(true);
        list.add(col);

        col = new ColumnConfig();
        col.setColumnType(ColumnType.N);
        col.setColumnName("B");
        col.setColumnNum(1);
        col.setFinalSelect(true);
        list.add(col);

        Scorer s = new Scorer(models, list, "NN", modelConfig);

        double[] input = { 0., 0., 3. };
        double[] ideal = { 1. };
        MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(ideal));

        Assert.assertEquals(s.score(pair, null).getScores().size(), 0);
    }

    @AfterClass
    public void delete() throws IOException {
        FileUtils.deleteDirectory(new File("tmp"));

        FileUtils.deleteDirectory(new File("models"));
        FileUtils.deleteDirectory(new File("test-output"));
        FileUtils.deleteDirectory(new File(Constants.COLUMN_META_FOLDER_NAME));
    }
}
