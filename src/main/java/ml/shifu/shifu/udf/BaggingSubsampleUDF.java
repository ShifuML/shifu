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
package ml.shifu.shifu.udf;

import java.io.IOException;
import java.util.Random;

import org.apache.pig.data.BagFactory;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;

/**
 * BaggingSubsampleUDF class
 */
public class BaggingSubsampleUDF extends AbstractTrainerUDF<DataBag> {
    private Random rand = new Random(System.currentTimeMillis());

    public BaggingSubsampleUDF(String source, String pathModelConfig, String pathColumnConfig) throws Exception {
        super(source, pathModelConfig, pathColumnConfig);
        log.debug("Bagging Options - Number of Models: " + modelConfig.getBaggingNum());
        log.debug("Bagging Options - Subsample Rate: " + modelConfig.getBaggingSampleRate());
    }

    public DataBag exec(Tuple input) throws IOException {
        int numBags = modelConfig.getBaggingNum();
        double rate = modelConfig.getBaggingSampleRate();

        DataBag bag = BagFactory.getInstance().newDefaultBag();
        for(int i = 0; i < numBags; i++) {
            double r = rand.nextDouble();
            if(r <= rate) {
                Tuple t = TupleFactory.getInstance().newTuple();
                t.append(i);
                t.append(input);
                bag.add(t);
            }
        }
        return bag;
    }

    public Schema outputSchema(Schema input) {
        return null;
    }

}
