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
import ml.shifu.shifu.container.obj.ModelNormalizeConf.NormType;

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.Arrays;


public class NormalizerTest {

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
        
        // Test norm value
        Double normValue1 = Normalizer.normalize(config, "5.0", 4.0, NormType.ZSCALE);
        Assert.assertEquals(normValue1, 3.0);
        
        Double normValue2 = Normalizer.normalize(config, "wrong_format", 4.0, NormType.ZSCALE);
        Assert.assertEquals(normValue2, 0.0);
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
        
        // Test norm value
        Double normValue1 = Normalizer.normalize(config, "3.0", 4.0, NormType.WEIGHT_WOE);
        Assert.assertEquals(normValue1, 3.2);
        
        Double normValue2 = Normalizer.normalize(config, "wrong_format", 4.0, NormType.WEIGHT_WOE);
        Assert.assertEquals(normValue2, 4.2);
        
        Double normValue3 = Normalizer.normalize(config, "3.0", 4.0, NormType.WOE);
        Assert.assertEquals(normValue3, 1.1);
        
        Double normValue4 = Normalizer.normalize(config, "wrong_format", 4.0, NormType.WOE);
        Assert.assertEquals(normValue4, 2.1);
    }
    
    @Test
    public void hybridNormalizeTest() {
        ColumnConfig config = new ColumnConfig();
        config.setMean(2.0);
        config.setStdDev(1.0);
        config.setColumnType(ColumnType.C);
        
        ColumnBinning cbin = new ColumnBinning();
        cbin.setBinCountWoe(Arrays.asList(new Double[]{1.1, 2.1, 3.1}));
        cbin.setBinWeightedWoe(Arrays.asList(new Double[]{3.2, 4.2, 5.2}));
        cbin.setBinCategory(Arrays.asList(new String[]{"a", "b"}));
        config.setColumnBinning(cbin);
        
        Double normValue1 = Normalizer.normalize(config, "a", 4.0, NormType.HYBRID);
        Assert.assertEquals(normValue1, 1.1);
        
        Double normValue2 = Normalizer.normalize(config, "wrong_format", 4.0, NormType.HYBRID);
        Assert.assertEquals(normValue2, 3.1);
        
        Double normValue3 = Normalizer.normalize(config, "b", 4.0, NormType.WEIGHT_HYBRID);
        Assert.assertEquals(normValue3, 4.2);
        
        Double normValue4 = Normalizer.normalize(config, "wrong_format", 4.0, NormType.WEIGHT_HYBRID);
        Assert.assertEquals(normValue4, 5.2);
        
        config.setColumnType(ColumnType.N);
        
        Double normValue5 = Normalizer.normalize(config, "5.0", 4.0, NormType.HYBRID);
        Assert.assertEquals(normValue5, 3.0);
        
        Double normValue6 = Normalizer.normalize(config, "wrong_format", 4.0, NormType.HYBRID);
        Assert.assertEquals(normValue6, 0.0);
        
        Double normValue7 = Normalizer.normalize(config, "5.0", 4.0, NormType.WEIGHT_HYBRID);
        Assert.assertEquals(normValue7, 3.0);
        
        Double normValue8 = Normalizer.normalize(config, "wrong_format", 4.0, NormType.WEIGHT_HYBRID);
        Assert.assertEquals(normValue8, 0.0);
        
    }
    
}
