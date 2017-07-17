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

import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.core.alg.NNTrainer;
import ml.shifu.shifu.util.Constants;

import org.apache.commons.io.FileUtils;
import org.encog.Encog;
import org.encog.engine.network.activation.*;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.ml.data.basic.BasicMLDataSet;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.propagation.Propagation;
import org.encog.neural.networks.training.propagation.quick.QuickPropagation;
import org.encog.persist.EncogDirectoryPersistence;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;


public class NNTrainerTest {

    private MLDataSet trainSet;
    private BasicNetwork network;

    private final static MLDataSet xor_Trainset = new BasicMLDataSet();
    //private final static Integer numberXorSet = 4 * 3;
    private final static MLDataSet xor_Validset = new BasicMLDataSet();

    static {
        double[] input = {0., 0.,};
        double[] ideal = {0.};
        MLDataPair pair = new BasicMLDataPair(new BasicMLData(input),
                new BasicMLData(ideal));
        xor_Trainset.add(pair);
        xor_Validset.add(pair);

        input = new double[]{0., 1.,};
        ideal = new double[]{1.};
        pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(
                ideal));
        xor_Trainset.add(pair);
        xor_Validset.add(pair);

        input = new double[]{1., 0.,};
        ideal = new double[]{1.};
        pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(
                ideal));
        xor_Trainset.add(pair);
        xor_Validset.add(pair);

        input = new double[]{1., 1.,};
        ideal = new double[]{0.};
        pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(
                ideal));
        xor_Trainset.add(pair);
        xor_Validset.add(pair);
    }

    @BeforeClass
    public void setUp() {
        trainSet = new BasicMLDataSet();
        network = new BasicNetwork();

        network.addLayer(new BasicLayer(new ActivationLinear(), true, 2));
        network.addLayer(new BasicLayer(new ActivationSigmoid(), true, 4));
        network.addLayer(new BasicLayer(new ActivationLOG(), true, 3));
        network.addLayer(new BasicLayer(new ActivationSIN(), true, 3));
        network.addLayer(new BasicLayer(new ActivationTANH(), false, 1));

        network.getStructure().finalizeStructure();
        network.reset();
    }

//    @Test
    public void testXorOperation() throws IOException {
        ModelConfig config = ModelConfig.createInitModelConfig(".", ALGORITHM.NN, ".", false);

        config.getTrain().setBaggingSampleRate(1.0);
        config.getTrain().setValidSetRate(0.1);
        config.getTrain().getParams().put("Propagation", "Q");
        config.getTrain().getParams().put("NumHiddenLayers", 1);
        config.getTrain().getParams().put("LearningRate", 1);
        List<Integer> nodes = new ArrayList<Integer>();
        nodes.add(5);
        List<String> func = new ArrayList<String>();
        func.add("tanh");
        config.getTrain().getParams().put("NumHiddenNodes", nodes);
        config.getTrain().getParams().put("ActivationFunc", func);
        config.getTrain().setNumTrainEpochs(100);

        NNTrainer trainer = new NNTrainer(config, 0, false);

        trainer.setTrainSet(xor_Trainset);
        trainer.setValidSet(xor_Validset);

        trainer.train();

        BasicNetwork bn = trainer.getNetwork();
        boolean[] cases = {true, false, false, true};
        int i = 0;
        for (MLDataPair data : xor_Validset) {
            double[] score = bn.compute(data.getInput()).getData();
            Assert.assertEquals(score[0] * 1000 < 500, cases[i]);
            i++;
        }
        Assert.assertEquals(bn.getLayerCount(), (Integer) (config.getTrain().getParams().get("NumHiddenLayers")) + 2 /*add input output*/);
    }

    @Test(expectedExceptions = RuntimeException.class)
    public void testExceptionWhileSetupModel() throws IOException {
        ModelConfig config = ModelConfig.createInitModelConfig(".", ALGORITHM.NN, ".", false);

        config.getTrain().getParams().put("Propagation", "Q");
        config.getTrain().getParams().put("NumHiddenLayers", 2);
        config.getTrain().getParams().put("LearningRate", 0.1);
        List<Integer> nodes = new ArrayList<Integer>();
        nodes.add(3);
        nodes.add(3);
        nodes.add(3);
        List<String> func = new ArrayList<String>();
        func.add("tanh");
        config.getTrain().getParams().put("NumHiddenNodes", nodes);
        config.getTrain().getParams().put("ActivationFunc", func);
        config.getTrain().setNumTrainEpochs(50);

        NNTrainer trainer = new NNTrainer(config, 0, false);
        try {
            trainer.setDataSet(xor_Trainset);
        } catch (IOException e) {

        }
        trainer.buildNetwork();
    }

    @Test
    public void testAndOperation() throws IOException {
        MLDataPair dataPair0 = BasicMLDataPair.createPair(2, 1);
        dataPair0.setInputArray(new double[]{0.0, 0.0});
        dataPair0.setIdealArray(new double[]{0.0});
        trainSet.add(dataPair0);

        MLDataPair dataPair1 = BasicMLDataPair.createPair(2, 1);
        dataPair1.setInputArray(new double[]{0.0, 1.0});
        dataPair1.setIdealArray(new double[]{0.0});
        trainSet.add(dataPair1);

        MLDataPair dataPair2 = BasicMLDataPair.createPair(2, 1);
        dataPair2.setInputArray(new double[]{1.0, 0.0});
        dataPair2.setIdealArray(new double[]{0.0});
        trainSet.add(dataPair2);

        MLDataPair dataPair3 = BasicMLDataPair.createPair(2, 1);
        dataPair3.setInputArray(new double[]{1.0, 1.0});
        dataPair3.setIdealArray(new double[]{1.0});
        trainSet.add(dataPair3);

        Propagation propagation = new QuickPropagation(network, trainSet, 0.1);

        double error = 0.0;
        double lastError = Double.MAX_VALUE;
        int iterCnt = 0;
        do {
            propagation.iteration();
            lastError = error;
            error = propagation.getError();
            System.out.println("The #" + (++iterCnt)
                    + " error is " + error);
        } while (Math.abs(lastError - error) > 0.001);

        propagation.finishTraining();

        File tmp = new File("model_folder");
        if (!tmp.exists()) {
            FileUtils.forceMkdir(tmp);
        }
        File modelFile = new File(
                "model_folder/model6.nn");
        EncogDirectoryPersistence.saveObject(modelFile, network);
        Assert.assertTrue(modelFile.exists());
        FileUtils.deleteQuietly(modelFile);
    }

    @Test
    public void testExistingModels() throws IOException {
        MLDataPair dataPair0 = BasicMLDataPair.createPair(2, 1);
        dataPair0.setInputArray(new double[]{-0.866025, -0.866025});
        dataPair0.setIdealArray(new double[]{0.0});
        trainSet.add(dataPair0);

        MLDataPair dataPair1 = BasicMLDataPair.createPair(2, 1);
        dataPair1.setInputArray(new double[]{-0.866025, 0.866025});
        dataPair1.setIdealArray(new double[]{0.0});
        trainSet.add(dataPair1);

        MLDataPair dataPair2 = BasicMLDataPair.createPair(2, 1);
        dataPair2.setInputArray(new double[]{0.866025, -0.866025});
        dataPair2.setIdealArray(new double[]{0.0});
        trainSet.add(dataPair2);

        MLDataPair dataPair3 = BasicMLDataPair.createPair(2, 1);
        dataPair3.setInputArray(new double[]{0.866025, 0.866025});
        dataPair3.setIdealArray(new double[]{1.0});
        trainSet.add(dataPair3);

        File modelDir = new File("model_folder");
        
        if (modelDir.isDirectory()) {
            File[] files = modelDir.listFiles();
            if (files != null) {
                for (File modelFile : files) {
                    System.out.println("result of " + modelFile.getName() + ":");
                    computeScore(modelFile, dataPair0, dataPair1, dataPair2, dataPair3);
                }
            } else {
                throw new IOException(String.format("Failed to list files in %s", modelDir.getAbsolutePath())); 
            }
        } else {
            System.err.println("No ./model_folder exist!");
        }
        
    }

    private void computeScore(File modelFile, MLDataPair dataPair0,
                              MLDataPair dataPair1, MLDataPair dataPair2, MLDataPair dataPair3) {
        BasicNetwork model = (BasicNetwork) EncogDirectoryPersistence
                .loadObject(modelFile);

        System.out.println((int) (model.compute(dataPair0.getInput())
                .getData(0) * 1000));
        System.out.println((int) (model.compute(dataPair1.getInput())
                .getData(0) * 1000));
        System.out.println((int) (model.compute(dataPair2.getInput())
                .getData(0) * 1000));
        System.out.println((int) (model.compute(dataPair3.getInput())
                .getData(0) * 1000));
    }

    @AfterClass
    public void shutDown() throws IOException {
        FileUtils.deleteDirectory(new File("./models/"));
        FileUtils.deleteDirectory(new File("./modelsTmp/"));
        FileUtils.deleteDirectory(new File("model_folder"));
        FileUtils.deleteDirectory(new File("tmp"));

        FileUtils.deleteQuietly(new File("init0.json"));
        FileUtils.deleteDirectory(new File(Constants.COLUMN_META_FOLDER_NAME));
        
        Encog.getInstance().shutdown();
    }
}
