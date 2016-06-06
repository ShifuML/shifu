/*
 * Copyright [2013-2015] PayPal Software Foundation
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
package ml.shifu.shifu.core.dtrain;

import java.io.File;
import java.util.Iterator;

import ml.shifu.shifu.core.dtrain.dataset.MemoryDiskMLDataSet;
import ml.shifu.shifu.util.SizeEstimator;

import org.apache.commons.io.FileUtils;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.basic.BasicMLData;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.testng.annotations.AfterClass;
import org.testng.annotations.Test;

public class MemoryDiskMLDataSetTest {

    private static double[] createInput(double d) {
        double[] input = new double[10];
        // Random r = new Random();
        for(int i = 0; i < input.length; i++) {
            input[i] = d;
        }
        return input;
    }
    
    @Test
    public void test(){
        double[] input = createInput(1);
        double[] output = new double[] { 1d };

        MLDataPair pair = new BasicMLDataPair(new BasicMLData(input), new BasicMLData(output));

        MemoryDiskMLDataSet dataSet = new MemoryDiskMLDataSet(400, "a.txt");
        dataSet.beginLoad(10, 1);
        dataSet.add(pair);
        MLDataPair pair2 = new BasicMLDataPair(new BasicMLData(createInput(2)), new BasicMLData(output));
        MLDataPair pair3 = new BasicMLDataPair(new BasicMLData(createInput(3)), new BasicMLData(output));
        MLDataPair pair4 = new BasicMLDataPair(new BasicMLData(createInput(4)), new BasicMLData(output));
        MLDataPair pair5 = new BasicMLDataPair(new BasicMLData(createInput(5)), new BasicMLData(output));
        MLDataPair pair6 = new BasicMLDataPair(new BasicMLData(createInput(6)), new BasicMLData(output));
        dataSet.add(pair2);
        dataSet.add(pair3);
        dataSet.add(pair4);
        dataSet.add(pair5);
        dataSet.add(pair6);
        dataSet.endLoad();
        long recordCount = dataSet.getRecordCount();
        for(long i = 0; i < recordCount; i++) {
            long start = System.currentTimeMillis();
            MLDataPair p = new BasicMLDataPair(new BasicMLData(createInput(6)), new BasicMLData(output));
            dataSet.getRecord(i, p);
            System.out.println((System.currentTimeMillis() - start) + " " + p);
        }

        System.out.println();

        Iterator<MLDataPair> iterator = dataSet.iterator();
        while(iterator.hasNext()) {
            long start = System.currentTimeMillis();
            MLDataPair next = iterator.next();
            System.out.println((System.currentTimeMillis() - start) + " " + next);
        }

        System.out.println();

        iterator = dataSet.iterator();
        while(iterator.hasNext()) {
            long start = System.currentTimeMillis();
            MLDataPair next = iterator.next();
            System.out.println((System.currentTimeMillis() - start) + " " + next);
        }

        dataSet.close();

        long size = SizeEstimator.estimate(pair);
        System.out.println(size);
    }

    @AfterClass
    public void cleanup() {
        FileUtils.deleteQuietly(new File("a.txt"));
    }
}
