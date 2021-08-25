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

import org.testng.Assert;
import org.testng.annotations.Test;

import java.util.HashSet;
import java.util.List;

/**
 * EqualIntervalBinningTest class
 *
 * @Oct 27, 2014
 */
public class EqualIntervalBinningTest {

    @Test
    public void test() {
        EqualIntervalBinning inst = new EqualIntervalBinning();
        inst.missingValSet = new HashSet<>();
        inst.expectedBinningNum = 5;
        inst.addData("");
        inst.addData(null);
        inst.addData(Double.toString(Double.MAX_VALUE));
        inst.addData("0.0");
        inst.addData("1.0");
        inst.addData("2.0");
        inst.addData("3.0");
        inst.addData("4.0");
        inst.addData("5.0");
        List<Double> bins = inst.getDataBin();
        Assert.assertTrue(bins.size() == 5);
        Assert.assertTrue(bins.get(1) > 0.5);
        Assert.assertTrue(bins.get(bins.size() - 1) < 4.5);
    }

}
