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

import java.util.Random;

import org.testng.annotations.Test;

/**
 * EqualIntervalBinningTest class
 * 
 * @Oct 27, 2014
 *
 */
public class EqualIntervalBinningTest {

    @Test
    public void testBinning() {
        Random rd = new Random(System.currentTimeMillis());
        
        EqualIntervalBinning binning = new EqualIntervalBinning(10);
        for ( int i = 0; i < 10000; i ++ ) {
            binning.addData(Integer.toString(rd.nextInt() % 1000));
        }
        
        System.out.println(binning.getDataBin());
    }
    
    
    @Test
    public void tesGussiantBinning() {
        Random rd = new Random(System.currentTimeMillis());
        
        EqualIntervalBinning binning = new EqualIntervalBinning(10);
        for ( int i = 0; i < 10000; i ++ ) {
            binning.addData(Double.toString(rd.nextGaussian() % 1000));
        }
        
        System.out.println(binning.getDataBin());
    }
    
}
