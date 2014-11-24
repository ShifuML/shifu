package ml.shifu.shifu.core.dvarsel.util;
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

import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.basic.BasicMLDataPair;
import org.encog.neural.networks.BasicNetwork;
import org.encog.util.concurrency.EngineTask;

/**
 * Created on 11/24/2014.
 */
public class MSEWorker implements EngineTask {
    private final BasicNetwork network;
    private final NNValidator owner;
    private final MLDataSet dataSet;
    private final int low;
    private final int high;
    private final MLDataPair pair;

    public MSEWorker(BasicNetwork network, NNValidator owner, MLDataSet dataSet, int low, int high) {
        this.network = network;
        this.owner = owner;
        this.dataSet = dataSet;
        this.low = low;
        this.high = high;
        this.pair = BasicMLDataPair.createPair(network.getInputCount(), network.getOutputCount());
    }

    public void run() {
        for (int i = this.low; i <= this.high; i++) {
            this.dataSet.getRecord(i, pair);
            MLData result = this.network.compute(this.pair.getInput());

            double tmp = result.getData()[0] - this.pair.getIdeal().getData()[0];
            double mse = tmp * tmp;

            owner.report(mse);
        }
    }
}
