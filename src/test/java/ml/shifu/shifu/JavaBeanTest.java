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
package ml.shifu.shifu;

import ml.shifu.shifu.container.*;
import ml.shifu.shifu.container.BinningObject.DataType;
import ml.shifu.shifu.container.BinningObject.VariableObjectComparator;
import ml.shifu.shifu.container.ModelResultObject.ModelResultObjectComparator;
import ml.shifu.shifu.container.ValueObject.ValueObjectComparator;
import ml.shifu.shifu.container.meta.MetaGroup;
import ml.shifu.shifu.container.meta.MetaItem;
import ml.shifu.shifu.container.meta.ValidateResult;
import ml.shifu.shifu.container.meta.ValueOption;
import ml.shifu.shifu.container.obj.*;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnConfigComparator;
import ml.shifu.shifu.container.obj.ModelTrainConf.ALGORITHM;
import ml.shifu.shifu.core.Binning.BinningDataType;
import ml.shifu.shifu.fs.SourceFile;
import ml.shifu.shifu.message.*;
import org.apache.commons.io.FileUtils;
import org.testng.Assert;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

import java.beans.IntrospectionException;
import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;


public class JavaBeanTest {

    @Test
    public void testAllJavaBeans() throws IntrospectionException, IOException {

        // JavaBeanTester.test(ScanStatsRawDataMessage.class);
        // JavaBeanTester.test(ScanTrainDataMessage.class);
        JavaBeanTester.test(AkkaActorInputMessage.class);
        JavaBeanTester.test(ColumnScoreMessage.class);
        JavaBeanTester.test(EvalResultMessage.class);
        JavaBeanTester.test(RunModelDataMessage.class);
        JavaBeanTester.test(RunModelResultMessage.class);
        // JavaBeanTester.test(ScanEvalDataMessage.class);
        JavaBeanTester.test(StatsPartRawDataMessage.class);
        JavaBeanTester.test(StatsValueObjectMessage.class);
        JavaBeanTester.test(TrainResultMessage.class);
        JavaBeanTester.test(TrainPartDataMessage.class);
        JavaBeanTester.test(NormPartRawDataMessage.class);
        JavaBeanTester.test(NormResultDataMessage.class);
        JavaBeanTester.test(StatsResultMessage.class);
        // JavaBeanTester.test(TrainInstanceMessage.class);

        JavaBeanTester.test(ColumnBinning.class);
        JavaBeanTester.test(ColumnStats.class);
//        JavaBeanTester.test(ColumnConfig.class);

        JavaBeanTester.test(ModelBasicConf.class);
        JavaBeanTester.test(ModelSourceDataConf.class);
        JavaBeanTester.test(ModelStatsConf.class);
        JavaBeanTester.test(ModelVarSelectConf.class);
        JavaBeanTester.test(ModelNormalizeConf.class);
        JavaBeanTester.test(ModelTrainConf.class);
        JavaBeanTester.test(ModelConfig.class);

        JavaBeanTester.test(ValueOption.class);
        JavaBeanTester.test(ValidateResult.class);
        JavaBeanTester.test(MetaItem.class);
        JavaBeanTester.test(MetaGroup.class);

//        JavaBeanTester.test(CaseScoreResult.class);
        JavaBeanTester.test(ModelResultObject.class);
        JavaBeanTester.test(PerformanceObject.class);
        JavaBeanTester.test(ReasonResultObject.class);
//        JavaBeanTester.test(ScoreObject.class);
        JavaBeanTester.test(VariableStoreObject.class);
        JavaBeanTester.test(ValueObject.class);
        JavaBeanTester.test(EvalConfig.class);
        JavaBeanTester.test(ModelInitInputObject.class);
        JavaBeanTester.test(WeightAmplifier.class);
        JavaBeanTester.test(ColumnScoreObject.class);
        JavaBeanTester.test(SourceFile.class);

        ModelResultObjectComparator modelResultObjectComparator = new ModelResultObjectComparator();
        modelResultObjectComparator.compare(new ModelResultObject(1, "2", 3d), new ModelResultObject(1, "2", 3d));

        ColumnConfigComparator cfc = new ColumnConfigComparator("KS");
        ColumnConfig columnConfig = new ColumnConfig();
        columnConfig.setKs(0.0d);
        columnConfig.setIv(0.0d);
        cfc.compare(columnConfig, columnConfig);
        cfc = new ColumnConfigComparator("IV");
        cfc.compare(columnConfig, columnConfig);

        HashMap<String, String> hashMap = new HashMap<String, String>();
        hashMap.put("id", "12");
        new ScoreObject(Arrays.asList(1d), 1);
        new ScoreObject(Arrays.asList(1d), 0);

        ValueObjectComparator voc = new ValueObjectComparator(BinningDataType.Categorical);
        ValueObject valueObject = new ValueObject();
        valueObject.setRaw("123");
        valueObject.setTag("1");
        valueObject.setValue(1.0d);
        voc.compare(valueObject, valueObject);

        ValueObject valueObject2 = new ValueObject();
        valueObject2.setRaw("345");
        valueObject2.setTag("1");
        valueObject2.setValue(2.0d);
        voc.compare(valueObject, valueObject2);
        voc.compare(valueObject, valueObject);
        voc = new ValueObjectComparator(BinningDataType.Numerical);

        voc.compare(valueObject, valueObject2);
        voc.compare(valueObject, valueObject);

        ModelConfig.createInitModelConfig("c", ALGORITHM.NN, "aaa", false);

        BinningObject bo = new BinningObject(DataType.Numerical);
        bo.getNumericalData();
        bo.getScore();
        bo.getTag();
        bo.getType();
        bo.setNumericalData(1d);
        bo.setScore(1.0d);
        bo.setTag("1");
        bo.toString();

        BinningObject bo2 = new BinningObject(DataType.Categorical);
        bo2.getCategoricalData();
        bo2.getScore();
        bo2.getTag();
        bo2.getType();
        bo2.setCategoricalData("111");
        bo2.setScore(1.0d);
        bo2.setTag("1");
        bo2.toString();

        BinningObject bo3 = new BinningObject(DataType.Numerical);
        bo3.getNumericalData();
        bo3.getScore();
        bo3.getTag();
        bo3.getType();
        bo3.setNumericalData(111d);
        bo3.setScore(1.0d);
        bo3.setTag("1");
        bo3.toString();

        BinningObject bo4 = new BinningObject(DataType.Categorical);
        bo4.getCategoricalData();
        bo4.getScore();
        bo4.getTag();
        bo4.getType();
        bo4.setCategoricalData("111");
        bo4.setScore(1.0d);
        bo4.setTag("1");
        bo4.toString();

        VariableObjectComparator vooc = new VariableObjectComparator();
        vooc.compare(bo, bo3);
        vooc.compare(bo2, bo4);

        ExceptionMessage es = new ExceptionMessage(new RuntimeException());
        es.setException(new RuntimeException());
        es.getException();
    }

    @Test(expectedExceptions = RuntimeException.class)
    public void binningObjectGetData() {
        BinningObject object = new BinningObject(DataType.Numerical);
        object.getCategoricalData();
    }

    @Test(expectedExceptions = RuntimeException.class)
    public void binningObjectSetData() {
        BinningObject object = new BinningObject(DataType.Numerical);
        object.setCategoricalData("test");
    }

    @Test
    public void binningObjectComparator() {
        BinningObject o1 = new BinningObject(DataType.Numerical);
        BinningObject o2 = new BinningObject(DataType.Numerical);

        o1.setNumericalData(0.1);
        o2.setNumericalData(0.2);

        o1.setTag("0");
        o2.setTag("1");

        VariableObjectComparator comp = new VariableObjectComparator();

        Assert.assertEquals(comp.compare(o1, o2), -1);

        o1 = new BinningObject(DataType.Categorical);
        o2 = new BinningObject(DataType.Categorical);

        o1.setCategoricalData("test1");
        o2.setCategoricalData("test2");
        o1.setTag("0");
        o2.setTag("1");

        Assert.assertEquals(comp.compare(o1, o2), -1);

    }

    @Test(expectedExceptions = RuntimeException.class)
    public void binningObjectComparatorException() {
        BinningObject o1 = new BinningObject(DataType.Numerical);
        BinningObject o2 = new BinningObject(DataType.Categorical);

        VariableObjectComparator comp = new VariableObjectComparator();

        comp.compare(o1, o2);
    }

    @AfterClass
    public void tearDown() throws IOException {
        FileUtils.deleteDirectory(new File("c"));
    }
}
