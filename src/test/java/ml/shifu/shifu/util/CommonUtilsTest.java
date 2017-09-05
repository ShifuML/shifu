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
package ml.shifu.shifu.util;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnFlag;
import ml.shifu.shifu.container.obj.ColumnType;
import ml.shifu.shifu.container.obj.EvalConfig;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.validator.ModelInspector;
import ml.shifu.shifu.fs.PathFinder;
import ml.shifu.shifu.udf.CalculateStatsUDF;
import ml.shifu.shifu.util.updater.ColumnConfigUpdater;

import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.hadoop.fs.FileStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

import com.fasterxml.jackson.core.JsonGenerationException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;

/**
 * CommonUtilsTest
 */
public class CommonUtilsTest {

    private static final Logger LOG = LoggerFactory.getLogger(CommonUtilsTest.class);

    private ObjectMapper jsonMapper = new ObjectMapper();

    @Test
    public void stringToIntegerListTest() {
        Assert.assertEquals(Arrays.asList(new Integer[] { 1, 2, 3 }), CommonUtils.stringToIntegerList("[1, 2, 3]"));
    }

    // @Test
    public void syncTest() throws IOException {
        ModelConfig config = ModelConfig.createInitModelConfig(".", ALGORITHM.NN, "test", false);
        config.setModelSetName("testModel");

        jsonMapper.writerWithDefaultPrettyPrinter().writeValue(new File("ModelConfig.json"), config);

        ColumnConfig col = new ColumnConfig();
        col.setColumnName("ColumnA");
        List<ColumnConfig> columnConfigList = new ArrayList<ColumnConfig>();
        columnConfigList.add(col);

        config.getDataSet().setSource(SourceType.LOCAL);;

        jsonMapper.writerWithDefaultPrettyPrinter().writeValue(new File("ColumnConfig.json"), columnConfigList);

        File file = null;
        file = new File("models");
        if(!file.exists()) {
            FileUtils.forceMkdir(file);
        }

        file = new File("models/model1.nn");
        if(!file.exists()) {
            if(file.createNewFile()) {
                BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file),
                        Constants.DEFAULT_CHARSET));
                writer.write("test string");
                writer.close();
            } else {
                LOG.warn("Create file {} failed", file.getAbsolutePath());
            }
        }

        file = new File("EvalSets/test");
        if(!file.exists()) {
            FileUtils.forceMkdir(file);
        }

        file = new File("EvalSets/test/EvalConfig.json");
        if(!file.exists()) {
            BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(file),
                    Constants.DEFAULT_CHARSET));
            writer.write("test string");
            writer.close();
        }

        CommonUtils.copyConfFromLocalToHDFS(config, new PathFinder(config));

        file = new File("ModelSets");
        Assert.assertTrue(file.exists());

        file = new File("ModelSets/testModel");
        Assert.assertTrue(file.exists());

        file = new File("ModelSets/testModel/ModelConfig.json");
        Assert.assertTrue(file.exists());

        file = new File("ModelSets/testModel/ColumnConfig.json");
        Assert.assertTrue(file.exists());

        file = new File("ModelSets/testModel/ReasonCodeMap.json");
        Assert.assertTrue(file.exists());

        file = new File("ModelSets/testModel/models/model1.nn");
        Assert.assertTrue(file.exists());

        file = new File("ModelSets/testModel/EvalSets/test/EvalConfig.json");
        Assert.assertTrue(file.exists());

        file = new File("ModelSets");
        if(file.exists()) {
            FileUtils.deleteDirectory(file);
        }

        file = new File("ColumnConfig.json");
        FileUtils.deleteQuietly(file);

        file = new File("ModelConfig.json");
        FileUtils.deleteQuietly(file);

        FileUtils.deleteDirectory(new File("models"));
        FileUtils.deleteDirectory(new File("EvalSets"));
    }

    // @Test
    public void syncUpEvalTest() throws IOException {
        ModelConfig config = ModelConfig.createInitModelConfig(".", ALGORITHM.NN, "test", false);
        config.setModelSetName("shifu");

        File file = new File("evals/EvalA");
        if(!file.exists()) {
            FileUtils.forceMkdir(file);
        }

        file = new File("testEval/EvalConfig.json");
        FileUtils.touch(file);

        // CommonUtils.copyEvalConfFromLocalToHDFS(config, "testEval");
        Assert.assertTrue(file.exists());

        FileUtils.deleteDirectory(new File("ModelSets"));
        FileUtils.deleteDirectory(new File("evals"));

    }

    @Test
    public void loadModelConfigTest() throws JsonGenerationException, JsonMappingException, IOException {
        ModelConfig config = ModelConfig.createInitModelConfig(".", ALGORITHM.NN, "test", false);
        config.setModelSetName("shifu");

        jsonMapper.writerWithDefaultPrettyPrinter().writeValue(new File("ModelConfig.json"), config);

        ModelConfig anotherConfig = CommonUtils.loadModelConfig();

        Assert.assertEquals(config, anotherConfig);

        FileUtils.deleteQuietly(new File("ModelConfig.json"));
    }

    @Test
    public void getFinalSelectColumnConfigListTest() {
        Collection<ColumnConfig> configList = new ArrayList<ColumnConfig>();

        ColumnConfig config = new ColumnConfig();
        config.setColumnName("A");
        config.setFinalSelect(false);

        configList.add(config);

        config = new ColumnConfig();
        config.setFinalSelect(true);
        config.setColumnName("B");

        configList.add(config);

        config = new ColumnConfig();
        config.setFinalSelect(false);
        config.setColumnName("C");
        configList.add(config);

        configList = CommonUtils.getFinalSelectColumnConfigList(configList);

        Assert.assertTrue(configList.size() == 1);

    }

    @Test
    public void getBinNumTest() {

        ColumnConfig config = new ColumnConfig();
        config.setColumnName("A");
        config.setColumnType(ColumnType.C);
        config.setBinCategory(Arrays.asList(new String[] { "2", "1", "3" }));

        int rt = CommonUtils.getBinNum(config, "2");

        Assert.assertTrue(rt == 0);

    }

    @Test
    public void testStringToIntegerList() {
        Assert.assertEquals(CommonUtils.stringToIntegerList("[]").size(), 1);
    }

    // @Test
    // public void assembleDataPairTest() throws Exception {
    // Map<String, String> rawDataMap = new HashMap<String, String>();
    // rawDataMap.put("ColumnA", "TestValue");
    //
    // ColumnConfig config = new ColumnConfig();
    // config.setColumnName("ColumnA");
    // List<ColumnConfig> columnConfigList = new ArrayList<ColumnConfig>();
    // columnConfigList.add(config);
    //
    // MLDataPair dp = CommonUtils.assembleDataPair(columnConfigList,
    // rawDataMap);
    // Assert.assertTrue(dp.getInput().getData().length == 0);
    //
    // Map<String, Object> objDataMap = new HashMap<String, Object>();
    // objDataMap.put("ColumnA", 10);
    // config.setFinalSelect(true);
    // config.setMean(12.0);
    // config.setStdDev(4.6);
    // MLDataPair pair = CommonUtils.assembleDataPair(columnConfigList,
    // objDataMap);
    // Assert.assertTrue(pair.getInput().getData()[0] < 0.0);
    // }

    @Test
    public void getTargetColumnNumTest() {
        List<ColumnConfig> list = new ArrayList<ColumnConfig>();
        ColumnConfig config = new ColumnConfig();
        config.setColumnFlag(null);

        list.add(config);

        config = new ColumnConfig();
        config.setColumnFlag(ColumnFlag.Target);
        config.setColumnNum(20);
        list.add(config);

        config = new ColumnConfig();
        config.setColumnFlag(null);
        list.add(config);

        Assert.assertEquals(Integer.valueOf(20), CommonUtils.getTargetColumnNum(list));
    }

    @Test
    public void loadModelsTest() {
        // TODO load models test
    }

    @Test
    public void getRawDataMapTest() {

        Map<String, String> map = CommonUtils.getRawDataMap(new String[] { "input1", "input2" }, new String[] { "1",
                "2" });

        Assert.assertTrue(map.containsKey("input2"));
        Assert.assertTrue(map.keySet().size() == 2);
    }

    @Test
    public void stringToDoubleListTest() {
        String str = "[0,1,2,3]";

        List<Integer> list = CommonUtils.stringToIntegerList(str);

        Assert.assertTrue(list.get(0) == 0);
    }

    // @Test
    public void updateColumnConfigFlagsTest() throws IOException {
        ModelConfig config = ModelConfig.createInitModelConfig("test", ALGORITHM.NN, "test", false);

        config.getDataSet().setMetaColumnNameFile("./conf/meta_column_conf.txt");
        config.getVarSelect().setForceRemoveColumnNameFile("./conf/remove_column_list.txt");
        List<ColumnConfig> list = new ArrayList<ColumnConfig>();

        ColumnConfig e = new ColumnConfig();
        e.setColumnName("a");
        list.add(e);

        e = new ColumnConfig();
        e.setColumnName("c");
        list.add(e);

        e = new ColumnConfig();
        e.setColumnName("d");
        list.add(e);

        ColumnConfigUpdater.updateColumnConfigFlags(config, list, ModelInspector.ModelStep.VARSELECT);

        Assert.assertTrue(list.get(0).isMeta());
    }

    @Test
    public void stringToStringListTest() {
        String str = "[1,2,3,,4]";

        List<Integer> list = CommonUtils.stringToIntegerList(str);

        Assert.assertTrue(list.get(0) == 1);
    }

    @Test
    public void getDerivedColumnNamesTest() {
        List<ColumnConfig> list = new ArrayList<ColumnConfig>();

        ColumnConfig e = new ColumnConfig();
        e.setColumnName("a");
        list.add(e);

        e = new ColumnConfig();
        e.setColumnName("derived_c");
        list.add(e);

        e = new ColumnConfig();
        e.setColumnName("d");
        list.add(e);

        List<String> output = CommonUtils.getDerivedColumnNames(list);

        Assert.assertEquals(output.get(0), "derived_c");
    }

    @Test
    public void testLoadModelConfig() throws IOException {
        ModelConfig config = CommonUtils.loadModelConfig(
                "src/test/resources/example/wdbc/wdbcModelSetLocal/ModelConfig.json", SourceType.LOCAL);
        Assert.assertEquals(config.getDataSet().getNegTags().get(0), "B");
    }

    @Test
    public void testEscape() {
        Assert.assertEquals("\\\\t", CommonUtils.escapePigString("\t"));
    }

    @AfterClass
    public void delete() throws IOException {
        FileUtils.deleteDirectory(new File("common-utils"));
    }

    @Test
    public void testFindModels() throws IOException {
        ModelConfig modelConfig = CommonUtils.loadModelConfig(
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json", SourceType.LOCAL);

        File srcModels = new File("src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/models");
        File dstModels = new File("models");
        FileUtils.copyDirectory(srcModels, dstModels);

        List<FileStatus> modelFiles = CommonUtils.findModels(modelConfig, null, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());

        EvalConfig evalConfig = modelConfig.getEvalConfigByName("EvalA");
        evalConfig.setCustomPaths(new HashMap<String, String>());
        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH, null);
        modelFiles = CommonUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH, "  ");
        modelFiles = CommonUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());

        FileUtils.deleteDirectory(dstModels);

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH,
                "./src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/models");
        modelFiles = CommonUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH,
                "./src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/models/model0.nn");
        modelFiles = CommonUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(1, modelFiles.size());

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH, "not-exists");
        modelFiles = CommonUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(0, modelFiles.size());

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH,
                "./src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/models/*.nn");
        modelFiles = CommonUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH,
                "./src/test/resources/example/cancer-judgement/ModelStore/ModelSet{0,1,9}/*/*.nn");
        modelFiles = CommonUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());
    }

    @Test
    public void testStringToArray() {
        String input = "[-37.075125208681136, 0.5043788517677587, 1.2588712402838798, 2.543219666931007, 4.896511355654414, 8.986345381526105, 17.06859410430839, 33.557046979865774, 73.27777777777777, 231.63698630136986, 100000.0]";

        List<Double> output = CommonUtils.stringToDoubleList(input);

        Assert.assertEquals(
                output,
                Arrays.asList(new Double[] { -37.075125208681136, 0.5043788517677587, 1.2588712402838798,
                        2.543219666931007, 4.896511355654414, 8.986345381526105, 17.06859410430839, 33.557046979865774,
                        73.27777777777777, 231.63698630136986, 100000.0 }));
    }

    @Test
    public void testCategoryVauleSepartor() {
        List<String> strList = new ArrayList<String>();
        strList.add("[Hello, Testing");
        strList.add("Haha, It's a testing]");

        String joinStr = StringUtils.join(strList, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR);
        List<String> recoverList = CommonUtils.stringToStringList(joinStr, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR);
        Assert.assertEquals(2, recoverList.size());
        Assert.assertEquals(strList.get(0).substring(1), recoverList.get(0));
        Assert.assertEquals(strList.get(1).substring(0, strList.get(1).length() - 1), recoverList.get(1));
    }

    @Test
    public void testSortFileNames() {
        File[] modelFiles = new File[5];
        modelFiles[0] = new File("model3.nn");
        modelFiles[1] = new File("model1.nn");
        modelFiles[2] = new File("model0.nn");
        modelFiles[3] = new File("model4.nn");
        modelFiles[4] = new File("model2.nn");

        Arrays.sort(modelFiles, new Comparator<File>() {
            @Override
            public int compare(File from, File to) {
                return from.getName().compareTo(to.getName());
            }
        });

        Assert.assertEquals(modelFiles[0].getName(), "model0.nn");
        Assert.assertEquals(modelFiles[4].getName(), "model4.nn");
    }

    @Test
    public void binIndexTest() {
        Double[] array = { Double.NEGATIVE_INFINITY, 2.1E-4, 0.00351, 0.01488, 0.02945, 0.0642, 0.11367, 0.22522,
                0.23977 };
        List<Double> binBoundary = Arrays.asList(array);

        Assert.assertEquals(CommonUtils.getBinIndex(binBoundary, 0.00350), 1);
        Assert.assertEquals(CommonUtils.getBinIndex(binBoundary, 0.00351), 2);
        Assert.assertEquals(CommonUtils.getBinIndex(binBoundary, 0.00353), 2);
        Assert.assertEquals(CommonUtils.getBinIndex(binBoundary, 0.0642), 5);
        Assert.assertEquals(CommonUtils.getBinIndex(binBoundary, 0.00010), 0);
        Assert.assertEquals(CommonUtils.getBinIndex(binBoundary, 5D), 8);

    }

    @Test
    public void trimNumber() {
        Assert.assertEquals(CommonUtils.trimTag("1000"), "1000");
        Assert.assertEquals(CommonUtils.trimTag("1.000"), "1");
        Assert.assertEquals(CommonUtils.trimTag("1.0"), "1");
        Assert.assertEquals(CommonUtils.trimTag("0.0"), "0");
        Assert.assertEquals(CommonUtils.trimTag("1."), "1");
        Assert.assertEquals(CommonUtils.trimTag("0.0000"), "0");
        Assert.assertEquals(CommonUtils.trimTag("1.03400"), "1.034");
        Assert.assertEquals(CommonUtils.trimTag("1.034001"), "1.034001");
        Assert.assertEquals(CommonUtils.trimTag("192.168.0.1"), "192.168.0.1");
        Assert.assertEquals(CommonUtils.trimTag("192.168.0.0"), "192.168.0.0");
        Assert.assertEquals(CommonUtils.trimTag(".0000"), "0");
        Assert.assertEquals(CommonUtils.trimTag(".00001"), "0.00001");
        Assert.assertEquals(CommonUtils.trimTag(".M0001"), ".M0001");
        Assert.assertEquals(CommonUtils.trimTag("M."), "M.");
        Assert.assertEquals(CommonUtils.trimTag(".L"), ".L");
        Assert.assertEquals(CommonUtils.trimTag(" .L  "), ".L");
        Assert.assertEquals(CommonUtils.trimTag(" "), "");
        Assert.assertEquals(CommonUtils.trimTag(null), "");
        Assert.assertEquals(CommonUtils.trimTag("1.0B"), "1.0B");
    }

    @AfterClass
    public void tearDown() throws IOException {
        FileUtils.deleteDirectory(new File(Constants.COLUMN_META_FOLDER_NAME));
    }

}
