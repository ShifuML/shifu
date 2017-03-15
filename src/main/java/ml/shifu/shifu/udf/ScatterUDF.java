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

import org.apache.pig.data.BagFactory;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;

import java.io.IOException;

/**
 * ScatterUDF class
 */
public class ScatterUDF extends AbstractTrainerUDF<DataBag> {

    public ScatterUDF(String source, String pathColumnConfig) throws Exception {
        super(source, pathColumnConfig);
    }

    public DataBag exec(Tuple input) throws IOException {
        if (input == null || input.size() == 0) {
            return null;
        }

        int size = input.size();
        DataBag bag = BagFactory.getInstance().newDefaultBag();
        Object score = input.get(size - 1);

        for (int i = 0; i < size - 1; i++) {
            if (columnConfigList.get(i).isFinalSelect()) {
                Tuple t = TupleFactory.getInstance().newTuple();
                t.append(i);
                t.append(input.get(i));
                t.append(score);
                bag.add(t);
            }
        }

        return bag;
    }

    public Schema outputSchema(Schema input) {
        return null;
    }
}
