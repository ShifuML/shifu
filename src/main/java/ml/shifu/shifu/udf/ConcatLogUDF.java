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

import org.apache.pig.EvalFunc;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;

/**
 * ConcatLogUDF class is used to concat logs
 */
public class ConcatLogUDF extends EvalFunc<Tuple> {

    public ConcatLogUDF() {
        // Empty
    }

    public Tuple exec(Tuple input) throws ExecException {
        TupleFactory tupleFactory = TupleFactory.getInstance();

        if (input == null || input.size() == 0) {
            return null;
        }

        DataBag logBag = (DataBag) input.get(0);

        StringBuilder s = new StringBuilder();

        for (Tuple tuple : logBag) {
            //String[] r = tuple.get(0).toString().split("=");
            //if (r.length == 2) {
            //	s.append(r[1]);
            //} else {
            s.append(tuple.get(0));
            //}

        }

        Tuple result = tupleFactory.newTuple();
        //result.append(input.get(0));
        String[] vars = s.toString().split(",", -1);

        // num of columns, for debug
        result.append(vars.length);

        for (String var : vars) {
            result.append(var);
        }

        return result;
    }

    public Schema outputSchema(Schema input) {
        //Utils.getSchemaFromString(schemaString)
        return null;
    }

}
