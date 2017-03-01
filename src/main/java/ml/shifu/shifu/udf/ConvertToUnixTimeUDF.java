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
import org.apache.pig.data.Tuple;
import org.apache.pig.impl.logicalLayer.schema.Schema;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.concurrent.TimeUnit;

/**
 * ConvertToUnixTimeUDF class is used to convert date into Unix time format (Long)
 */
public class ConvertToUnixTimeUDF extends EvalFunc<Long> {

    private SimpleDateFormat sdf;

    public ConvertToUnixTimeUDF() {
        sdf = new SimpleDateFormat("MM/dd/yyyy HH:mm:ss");
    }

    public ConvertToUnixTimeUDF(String formatString) throws Exception {
        sdf = new SimpleDateFormat(formatString);
    }

    public Long exec(Tuple input) throws ExecException {
        if (input == null) {
            return null;
        }

        Date date;
        try {
            date = sdf.parse(input.get(0).toString());
        } catch (ParseException e) {
            e.printStackTrace();
            return 0L;
        }
        return TimeUnit.MILLISECONDS.toSeconds(date.getTime());
    }

    public Schema outputSchema(Schema input) {
        // Utils.getSchemaFromString(schemaString)
        return null;
    }

}
