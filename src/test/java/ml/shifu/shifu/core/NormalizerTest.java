/**
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

import ml.shifu.shifu.container.obj.ColumnBinning;
import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ColumnConfig.ColumnType;
import ml.shifu.shifu.container.obj.ModelNormalizeConf.MissValueFillType;

import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.util.Arrays;


public class NormalizerTest {

    @BeforeClass
    public void setUp() {

    }

    @Test
    public void computeZScore() {
        Assert.assertEquals(0.0, Normalizer.computeZScore(2, 2, 1, 6.0));
        Assert.assertEquals(6.0, Normalizer.computeZScore(12, 2, 1, 6.0));

        // If stdDev == 0, return 0
        Assert.assertEquals(0.0, Normalizer.computeZScore(12, 2, 0, 6.0));
    }

    @Test
    public void getZScore1() {
        ColumnConfig config = new ColumnConfig();
        config.setMean(2.0);
        config.setStdDev(1.0);
        config.setColumnType(ColumnType.N);
        Assert.assertEquals(0.0, Normalizer.normalize(config, "2", 6.0));

        Assert.assertEquals(0.0, Normalizer.normalize(config, "ABC", 0.1));
    }

    @Test
    public void getZScore2() {
        ColumnConfig config = new ColumnConfig();
        config.setMean(2.0);
        config.setStdDev(1.0);
        config.setColumnType(ColumnType.N);
        Assert.assertEquals(-4.0, Normalizer.normalize(config, "-3", null));
    }

    @Test
    public void getZScore3() {
        ColumnConfig config = new ColumnConfig();
        config.setColumnType(ColumnType.C);
        config.setMean(2.0);
        config.setStdDev(1.0);
        config.setBinCategory(Arrays.asList(new String[]{"1", "2", "3", "4", "ABC"}));
        config.setBinPosCaseRate(Arrays.asList(new Double[]{0.1, 2.0, 0.3, 0.1}));
        Assert.assertEquals(0.0, Normalizer.normalize(config, "2", 0.1));

        Assert.assertEquals(0.0, Normalizer.normalize(config, "5", 0.1));

    }

    @Test
    public void getZScore4() {
        ColumnConfig config = new ColumnConfig();
        Normalizer n = new Normalizer(config, 0.1);
        config.setMean(2.0);
        config.setStdDev(1.0);
        config.setColumnType(ColumnType.N);
        Assert.assertEquals(0.0, n.normalize("2"));
    }
    
    @Test
    public void zScoreNormalizeTest() {
        ColumnConfig config = new ColumnConfig();
        config.setMean(2.0);
        config.setStdDev(1.0);
        config.setColumnType(ColumnType.N);
        
        ColumnBinning cbin = new ColumnBinning();
        cbin.setBinCountWoe(Arrays.asList(new Double[]{1.1, 2.1}));
        cbin.setBinWeightedWoe(Arrays.asList(new Double[]{3.2, 4.2}));
        config.setColumnBinning(cbin);
        
        // Test miss value.
        Double missZero = Normalizer.zScoreNormalize(config, "wrongFormat", 4.0, MissValueFillType.ZERO);
        Assert.assertEquals(missZero, 0.0);
        
        Double missMean = Normalizer.zScoreNormalize(config, "wrongFormat", 4.0, MissValueFillType.MEAN);
        Assert.assertEquals(missMean, 2.0);
        
        Double missCWoe = Normalizer.zScoreNormalize(config, "wrongFormat", 4.0, MissValueFillType.COUNTWOE);
        Assert.assertEquals(missCWoe, 2.1);
        
        Double missWWoe = Normalizer.zScoreNormalize(config, "wrongFormat", 4.0, MissValueFillType.WEIGHTEDWOE);
        Assert.assertEquals(missWWoe, 4.2);
        
        // Test norm value
        Double normValue1 = Normalizer.zScoreNormalize(config, "5.0", 4.0, MissValueFillType.ZERO);
        Assert.assertEquals(normValue1, 3.0);
    }
    
    @Test
    public void woeNormalizeTest() {
        ColumnConfig config = new ColumnConfig();
        config.setMean(2.0);
        config.setStdDev(1.0);
        config.setColumnType(ColumnType.N);
        
        ColumnBinning cbin = new ColumnBinning();
        cbin.setBinCountWoe(Arrays.asList(new Double[]{1.1, 2.1}));
        cbin.setBinWeightedWoe(Arrays.asList(new Double[]{3.2, 4.2}));
        cbin.setBinBoundary(Arrays.asList(new Double[]{Double.NEGATIVE_INFINITY, 4.0}));
        config.setColumnBinning(cbin);
        
        // Test miss value.
        Double missZero = Normalizer.woeNormalize(config, "wrongFormat", true, MissValueFillType.ZERO);
        Assert.assertEquals(missZero, 0.0);
        
        Double missMean = Normalizer.woeNormalize(config, "wrongFormat", true, MissValueFillType.MEAN);
        Assert.assertEquals(missMean, 2.0);
        
        Double missCWoe = Normalizer.woeNormalize(config, "wrongFormat", true, MissValueFillType.COUNTWOE);
        Assert.assertEquals(missCWoe, 2.1);
        
        Double missWWoe = Normalizer.woeNormalize(config, "wrongFormat", true, MissValueFillType.WEIGHTEDWOE);
        Assert.assertEquals(missWWoe, 4.2);
        
        // Test norm value
        Double normValue1 = Normalizer.woeNormalize(config, "3.0", true, MissValueFillType.ZERO);
        Assert.assertEquals(normValue1, 3.2);
        
        Double normValue2 = Normalizer.woeNormalize(config, "3.0", false, MissValueFillType.ZERO);
        Assert.assertEquals(normValue2, 1.1);
    }
    
}
