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
        Assert.assertEquals(-2.0, Normalizer.computeZScore(2, 4, 1, 2));
        
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
    public void numericalNormalizeTest() {
        // Input setting
        ColumnConfig config = new ColumnConfig();
        config.setMean(2.0);
        config.setStdDev(1.0);
        config.setColumnType(ColumnType.N);
        
        ColumnBinning cbin = new ColumnBinning();
        cbin.setBinCountWoe(Arrays.asList(new Double[]{10.0, 11.0, 12.0, 13.0}));
        cbin.setBinWeightedWoe(Arrays.asList(new Double[]{20.0, 21.0, 22.0, 23.0}));
        cbin.setBinBoundary(Arrays.asList(new Double[]{Double.NEGATIVE_INFINITY, 2.0, 4.0}));
        config.setColumnBinning(cbin);
        
        // Test zscore normalization
        Assert.assertEquals(Normalizer.normalize(config, "5.0", 4.0, NormType.ZSCALE), 3.0);
        Assert.assertEquals(Normalizer.normalize(config, "5.0", null, NormType.ZSCALE), 3.0);
        Assert.assertEquals(Normalizer.normalize(config, null, 4.0, NormType.ZSCALE), 0.0);
        Assert.assertEquals(Normalizer.normalize(config, "wrong_format", 4.0, NormType.ZSCALE), 0.0);
        
        // Test woe normalization
        Assert.assertEquals(Normalizer.normalize(config, "3.0", null, NormType.WEIGHT_WOE), 21.0);
        Assert.assertEquals(Normalizer.normalize(config, "wrong_format", null, NormType.WEIGHT_WOE), 23.0);
        Assert.assertEquals(Normalizer.normalize(config, "3.0", null, NormType.WOE), 11.0);
        Assert.assertEquals(Normalizer.normalize(config, "wrong_format", null, NormType.WOE), 13.0);
        
        // Test hybrid normalization, for numerical use zscore.
        Assert.assertEquals(Normalizer.normalize(config, "5.0", 4.0, NormType.HYBRID), 3.0);
        Assert.assertEquals(Normalizer.normalize(config, "5.0", null, NormType.HYBRID), 3.0);
        Assert.assertEquals(Normalizer.normalize(config, null, 4.0, NormType.HYBRID), 0.0);
        Assert.assertEquals(Normalizer.normalize(config, "wrong_format", 4.0, NormType.HYBRID), 0.0);

        // Currently WEIGHT_HYBRID and HYBRID act same for numerical value.
        Assert.assertEquals(Normalizer.normalize(config, "5.0", 4.0, NormType.WEIGHT_HYBRID), 3.0);
        Assert.assertEquals(Normalizer.normalize(config, "5.0", null, NormType.WEIGHT_HYBRID), 3.0);
        Assert.assertEquals(Normalizer.normalize(config, null, 4.0, NormType.WEIGHT_HYBRID), 0.0);
        Assert.assertEquals(Normalizer.normalize(config, "wrong_format", 4.0, NormType.WEIGHT_HYBRID), 0.0);
    }
    
    @Test
    public void categoricalNormalizeTest() {
        // Input setting
        ColumnConfig config = new ColumnConfig();
        config.setMean(0.2);
        config.setStdDev(1.0);
        config.setColumnType(ColumnType.C);
        
        ColumnBinning cbin = new ColumnBinning();
        cbin.setBinCountWoe(Arrays.asList(new Double[]{10.0, 11.0, 12.0, 13.0}));
        cbin.setBinWeightedWoe(Arrays.asList(new Double[]{20.0, 21.0, 22.0, 23.0}));
        cbin.setBinCategory(Arrays.asList(new String[]{"a", "b", "c"}));
        cbin.setBinPosRate(Arrays.asList(new Double[]{0.2, 0.4, 0.8}));
        config.setColumnBinning(cbin);
        
        // Test zscore normalization
        Assert.assertEquals(Normalizer.normalize(config, "b", 4.0, NormType.ZSCALE), 0.2);
        Assert.assertEquals(Normalizer.normalize(config, "b", null, NormType.ZSCALE), 0.2);
        Assert.assertEquals(Normalizer.normalize(config, null, 4.0, NormType.ZSCALE), 0.0);
        Assert.assertEquals(Normalizer.normalize(config, "wrong_format", 4.0, NormType.ZSCALE), 0.0);
        
        // Test woe normalization
        Assert.assertEquals(Normalizer.normalize(config, "c", null, NormType.WEIGHT_WOE), 22.0);
        Assert.assertEquals(Normalizer.normalize(config, null, null, NormType.WEIGHT_WOE), 23.0);
        Assert.assertEquals(Normalizer.normalize(config, "wrong_format", null, NormType.WEIGHT_WOE), 23.0);
        Assert.assertEquals(Normalizer.normalize(config, "c", null, NormType.WOE), 12.0);
        Assert.assertEquals(Normalizer.normalize(config, null, null, NormType.WOE), 13.0);
        Assert.assertEquals(Normalizer.normalize(config, "wrong_format", null, NormType.WOE), 13.0);
        
        // Test hybrid normalization, for categorical value use [weight]woe.
        Assert.assertEquals(Normalizer.normalize(config, "a", null, NormType.HYBRID), 10.0);
        Assert.assertEquals(Normalizer.normalize(config, null, null, NormType.HYBRID), 13.0);
        Assert.assertEquals(Normalizer.normalize(config, "wrong_format", null, NormType.HYBRID), 13.0);
        
        Assert.assertEquals(Normalizer.normalize(config, "a", null, NormType.WEIGHT_HYBRID), 20.0);
        Assert.assertEquals(Normalizer.normalize(config, null, null, NormType.WEIGHT_HYBRID), 23.0);
        Assert.assertEquals(Normalizer.normalize(config, "wrong_format", null, NormType.WEIGHT_HYBRID), 23.0);
    }
    
}
