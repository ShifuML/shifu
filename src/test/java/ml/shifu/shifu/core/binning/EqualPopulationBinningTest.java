/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.binning;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.testng.Assert;
import org.testng.annotations.Test;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnType;
import ml.shifu.shifu.container.obj.ModelConfig;
import ml.shifu.shifu.container.obj.ModelStatsConf.BinningMethod;
import ml.shifu.shifu.util.Constants;

/**
 * EqualPopulationBinningTest class
 * 
 * @Oct 22, 2014
 *
 */
public class EqualPopulationBinningTest {

    @Test
    void testExtraSmallBins() {
        EqualPopulationBinning binning = new EqualPopulationBinning(10);
        for(int i = 0; i < 100000; i++) {
            binning.addData(5.0);
        }

        for(int i = 0; i < 100000; i++) {
            binning.addData(8.0);
        }

        for(int i = 0; i < 60; i++) {
            binning.addData(-10.0);
        }

        System.out.println(binning.getDataBin());
        Assert.assertEquals(2, binning.getDataBin().size());

        binning = new EqualPopulationBinning(10);
        for(int i = 0; i < 100000; i++) {
            binning.addData(5.0);
        }

        for(int i = 0; i < 100000; i++) {
            binning.addData(8.0);
        }

        for(int i = 0; i < 61; i++) { // more then threshold
            binning.addData(-10.0);
        }

        System.out.println(binning.getDataBin());
        Assert.assertEquals(3, binning.getDataBin().size());
    }

    @Test
    public void testBinning() {
        Random rd = new Random(System.currentTimeMillis());

        EqualPopulationBinning binning = new EqualPopulationBinning(10);
        long start = System.currentTimeMillis();
        for(int i = 0; i < 100000; i++) {
            binning.addData(Integer.toString(rd.nextInt() % 1000));
        }
        long end = System.currentTimeMillis();
        System.out.println("Spend " + (end - start) + " milli-seconds to create data.");
        System.out.println(binning.getDataBin());

        String binStr = binning.objToString();
        String[] fieldArr = binStr.split(Character.toString(AbstractBinning.FIELD_SEPARATOR));
        Assert.assertTrue(fieldArr.length == 6);
    }

    @Test
    public void tesGussiantBinning() {
        long startTs = System.currentTimeMillis();
        Random rd = new Random(System.currentTimeMillis());

        EqualPopulationBinning binning = new EqualPopulationBinning(10);
        for(int i = 0; i < 1000; i++) {
            binning.addData(rd.nextGaussian() % 1000, rd.nextDouble() * 5.0);
        }

        System.out.println(binning.getDataBin());
        System.out.println("spend " + (System.currentTimeMillis() - startTs) + " milliseconds to generate binnig.");
    }

    @Test
    public void testObjectSeri() {
        Random rd = new Random(System.currentTimeMillis());

        EqualPopulationBinning binning = new EqualPopulationBinning(10);
        for(int i = 0; i < 10000; i++) {
            binning.addData(Double.toString(rd.nextGaussian() % 1000));
        }

        String binningStr = binning.objToString();
        String originalBinningData = binning.getDataBin().toString();

        ModelConfig modelConfig = new ModelConfig();
        modelConfig.getStats().setBinningMethod(BinningMethod.EqualPositive);

        ColumnConfig columnConfig = new ColumnConfig();
        columnConfig.setColumnType(ColumnType.N);

        AbstractBinning<?> otherBinning = AbstractBinning.constructBinningFromStr(modelConfig, columnConfig, binningStr,
                Constants.MAX_CATEGORICAL_BINC_COUNT);
        String newBinningData = otherBinning.getDataBin().toString();

        Assert.assertEquals(originalBinningData, newBinningData);
    }

    @Test
    public void testBingMerge() {
        List<EqualPopulationBinning> binningList = new ArrayList<EqualPopulationBinning>();
        long start, end;

        for(int i = 0; i < 10; i++) {
            start = System.currentTimeMillis();
            binningList.add(createBinning());
            end = System.currentTimeMillis();
            System.out.println("Spend " + (end - start) + " milli-seconds to create " + i + "-th binning.");
        }

        EqualPopulationBinning binning = createBinning();
        for(int i = 0; i < 10; i++) {
            EqualPopulationBinning another = binningList.get(i);

            start = System.currentTimeMillis();
            binning.mergeBin(another);
            end = System.currentTimeMillis();
            Assert.assertTrue((end - start) < 1000);
        }

        System.out.println(binning.getDataBin().toString());
    }

    /**
     * @return
     */
    private EqualPopulationBinning createBinning() {
        Random rd = new Random(System.currentTimeMillis());

        EqualPopulationBinning binning = new EqualPopulationBinning(20);
        for(int i = 0; i < 18000; i++) {
            binning.addData(Double.toString(rd.nextDouble() % 1000));
        }

        return binning;
    }

    @Test
    public void testSerialObject() {
        EqualPopulationBinning binning = new EqualPopulationBinning(10);
        String binStr = binning.objToString();
        String[] fieldArr = binStr.split(Character.toString(AbstractBinning.FIELD_SEPARATOR));
        Assert.assertTrue(fieldArr.length == 5);
    }

}
