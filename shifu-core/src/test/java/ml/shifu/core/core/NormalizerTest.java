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
package ml.shifu.core.core;

import ml.shifu.core.container.obj.ColumnConfig;
import ml.shifu.core.container.obj.ColumnConfig.ColumnType;
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
}
