/**
 * Copyright [2012-2014] eBay Software Foundation
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
package ml.shifu.core.util;

import com.fasterxml.jackson.core.JsonGenerationException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import ml.shifu.core.container.obj.ColumnConfig;
import ml.shifu.core.container.obj.ColumnConfig.ColumnFlag;
import ml.shifu.core.container.obj.ColumnConfig.ColumnType;
import ml.shifu.core.container.obj.EvalConfig;
import ml.shifu.core.container.obj.ModelConfig;
import ml.shifu.core.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.core.container.obj.RawSourceData.SourceType;
import org.apache.commons.io.FileUtils;
import org.apache.hadoop.fs.FileStatus;
import org.encog.ml.data.MLDataPair;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;

/**
 * CommonUtilsTest
 */
public class CommonUtilsTest {

    private ObjectMapper jsonMapper = new ObjectMapper();

    @Test
    public void stringToIntegerListTest() {
        Assert.assertEquals(Arrays.asList(new Integer[]{1, 2, 3}),
                CommonUtils.stringToIntegerList("[1, 2, 3]"));
    }

    //@Test
    public void syncTest() throws IOException {
        ModelConfig config = ModelConfig
                .createInitModelConfig(".", ALGORITHM.NN, "test");
        config.setModelSetName("testModel");

        jsonMapper.writerWithDefaultPrettyPrinter().writeValue(
                new File("ModelConfig.json"), config);

        ColumnConfig col = new ColumnConfig();
        col.setColumnName("ColumnA");
        List<ColumnConfig> columnConfigList = new ArrayList<ColumnConfig>();
        columnConfigList.add(col);

        config.getDataSet().setSource(SourceType.LOCAL);
        ;

        jsonMapper.writerWithDefaultPrettyPrinter().writeValue(
                new File("ColumnConfig.json"), columnConfigList);

        File file = null;
        file = new File("models");
        if (!file.exists()) {
            file.mkdir();
        }

        file = new File("models/model1.nn");
        if (!file.exists()) {
            file.createNewFile();
            FileWriter write = new FileWriter(file);
            write.write("test string");
            write.close();
        }

        file = new File("EvalSets/test");
        if (!file.exists()) {
            file.mkdirs();
        }

        file = new File("EvalSets/test/EvalConfig.json");
        if (!file.exists()) {
            file.createNewFile();
            FileWriter write = new FileWriter(file);
            write.write("test string");
            write.close();
        }

        CommonUtils.copyConfFromLocalToHDFS(config);

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
        if (file.exists()) {
            FileUtils.deleteDirectory(file);
        }

        file = new File("ColumnConfig.json");
        file.delete();

        file = new File("ModelConfig.json");
        file.delete();

        FileUtils.deleteDirectory(new File("models"));
        FileUtils.deleteDirectory(new File("EvalSets"));
    }

    //@Test
    public void syncUpEvalTest() throws IOException {
        ModelConfig config = ModelConfig.createInitModelConfig(".", ALGORITHM.NN, "test");
        config.setModelSetName("core");

        File file = new File("evals/Eval1");
        if (!file.exists()) {
            file.mkdirs();
        }

        file = new File("testEval/EvalConfig.json");
        file.createNewFile();

        //CommonUtils.copyEvalConfFromLocalToHDFS(config, "testEval");
        Assert.assertTrue(file.exists());

        FileUtils.deleteDirectory(new File("ModelSets"));
        FileUtils.deleteDirectory(new File("evals"));

    }

    @Test
    public void loadModelConfigTest() throws JsonGenerationException,
            JsonMappingException, IOException {
        ModelConfig config = ModelConfig.createInitModelConfig(".", ALGORITHM.NN, "test");
        config.setModelSetName("core");

        jsonMapper.writerWithDefaultPrettyPrinter().writeValue(
                new File("ModelConfig.json"), config);

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
        config.setBinCategory(Arrays.asList(new String[]{"2", "1", "3"}));

        int rt = CommonUtils.getBinNum(config, "2");

        Assert.assertTrue(rt == 0);

    }

    @Test
    public void testStringToIntegerList() {
        Assert.assertEquals(CommonUtils.stringToIntegerList("[]").size(), 1);
    }

    @Test
    public void assembleDataPairTest() throws Exception {
        Map<String, String> rawDataMap = new HashMap<String, String>();
        rawDataMap.put("ColumnA", "TestValue");

        ColumnConfig config = new ColumnConfig();
        config.setColumnName("ColumnA");
        List<ColumnConfig> columnConfigList = new ArrayList<ColumnConfig>();
        columnConfigList.add(config);

        MLDataPair dp = CommonUtils.assembleDataPair(columnConfigList,
                rawDataMap);
        Assert.assertTrue(dp.getInput().getData().length == 0);

        Map<String, Object> objDataMap = new HashMap<String, Object>();
        objDataMap.put("ColumnA", 10);
        config.setFinalSelect(true);
        config.setMean(12.0);
        config.setStdDev(4.6);
        MLDataPair pair = CommonUtils.assembleDataPair(columnConfigList,
                objDataMap);
        Assert.assertTrue(pair.getInput().getData()[0] < 0.0);
    }

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
        //TODO load models test
    }

    @Test
    public void getRawDataMapTest() {

        Map<String, String> map = CommonUtils.getRawDataMap(new String[]{"input1", "input2"}, new String[]{"1", "2"});

        Assert.assertTrue(map.containsKey("input2"));
        Assert.assertTrue(map.keySet().size() == 2);
    }

    @Test
    public void stringToDoubleListTest() {
        String str = "[0,1,2,3]";

        List<Integer> list = CommonUtils.stringToIntegerList(str);

        Assert.assertTrue(list.get(0) == 0);
    }

    //@Test
    public void updateColumnConfigFlagsTest() throws IOException {
        ModelConfig config = ModelConfig.createInitModelConfig("test", ALGORITHM.NN, "test");

        config.getDataSet().setMetaColumnNameFile("./conf/meta_column_conf.txt");
        ;
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

        CommonUtils.updateColumnConfigFlags(config, list);

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
        ModelConfig config = CommonUtils.loadModelConfig("src/test/resources/example/wdbc/wdbcModelSetLocal/ModelConfig.json", SourceType.LOCAL);
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
        ModelConfig modelConfig = CommonUtils.loadModelConfig("src/test/resources/unittest/ModelSets/full/ModelConfig.json", SourceType.LOCAL);

        File srcModels = new File("src/test/resources/unittest/ModelSets/full/models");
        File dstModels = new File("models");
        FileUtils.copyDirectory(srcModels, dstModels);

        List<FileStatus> modelFiles = CommonUtils.findModels(modelConfig, null, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());

        EvalConfig evalConfig = modelConfig.getEvalConfigByName("Eval1");
        evalConfig.setCustomPaths(new HashMap<String, String>());
        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH, null);
        modelFiles = CommonUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH, "  ");
        modelFiles = CommonUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());

        FileUtils.deleteDirectory(dstModels);

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH, "./src/test/resources/unittest/ModelSets/full/models");
        modelFiles = CommonUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH, "./src/test/resources/unittest/ModelSets/full/models/model1.nn");
        modelFiles = CommonUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(1, modelFiles.size());

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH, "not-exists");
        modelFiles = CommonUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(0, modelFiles.size());

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH, "./src/test/resources/unittest/ModelSets/full/models/*.nn");
        modelFiles = CommonUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());

        evalConfig.getCustomPaths().put(Constants.KEY_MODELS_PATH, "./src/test/resources/example/cancer-judgement/ModelStore/ModelSet{0,1,9}/*/*.nn");
        modelFiles = CommonUtils.findModels(modelConfig, evalConfig, SourceType.LOCAL);
        Assert.assertEquals(5, modelFiles.size());
    }

    @AfterClass
    public void tearDown() {
        FileUtils.deleteQuietly(new File(Constants.DEFAULT_META_COLUMN_FILE));
        FileUtils.deleteQuietly(new File(Constants.DEFAULT_CATEGORICAL_COLUMN_FILE));
        FileUtils.deleteQuietly(new File(Constants.DEFAULT_FORCESELECT_COLUMN_FILE));
        FileUtils.deleteQuietly(new File(Constants.DEFAULT_FORCEREMOVE_COLUMN_FILE));
        FileUtils.deleteQuietly(new File("Eval1" + Constants.DEFAULT_EVALSCORE_META_COLUMN_FILE));
    }
}
