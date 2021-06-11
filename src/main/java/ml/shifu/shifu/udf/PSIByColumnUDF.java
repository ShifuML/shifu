/*
 * Copyright [2013-2021] PayPal Software Foundation
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
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.lang3.StringUtils;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;

/**
 * Calculate the Population Stability Index based on column and psiColumn values.
 */
public class PSIByColumnUDF extends AbstractTrainerUDF<Tuple> {

    public PSIByColumnUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
    }

    @Override
    public Tuple exec(Tuple input) throws IOException {
        if(input == null || input.size() < 2) {
            return null;
        }

        Integer columnId = (Integer) input.get(0);
        DataBag databag = (DataBag) input.get(1);

        List<Tuple> counters = new ArrayList<>();
        Iterator<Tuple> iter = databag.iterator();
        while(iter.hasNext()) {
            Tuple tuple = iter.next();
            if(tuple == null || tuple.size() == 0) {
                continue;
            }

            Tuple psiTuple = TupleFactory.getInstance().newTuple(3);
            String unit = tuple.get(3) == null ? "" : tuple.get(3).toString();
            psiTuple.set(0, unit); // psi column value

            String[] subBinArr = StringUtils.split((String) tuple.get(1), CalculateStatsUDF.CATEGORY_VAL_SEPARATOR);
            List<Double> subCounters = new ArrayList<Double>();
            double total = 0d;
            for(String binningElement: subBinArr) {
                double dVal = Double.parseDouble(binningElement);
                subCounters.add(dVal);
                total += dVal;
            }
            psiTuple.set(1, subCounters); // counters
            psiTuple.set(2, total);
            counters.add(psiTuple);
        }

        Collections.sort(counters, new Comparator<Tuple>() {
            @Override
            public int compare(Tuple t1, Tuple t2) {
                try {
                    String s1 = (String) t1.get(0);
                    String s2 = (String) t2.get(0);
                    return s1.compareTo(s2);
                } catch (ExecException e) {
                    log.error("Error:", e);
                }
                return 0;
            }
        });

        if(counters.size() <= 1) {
            log.warn("Column " + columnId + " with psi column value " + counters.size()
                    + " <=1, no need psi compute or please check psiColumn set in ModelConfig.json.");
        }

        StringBuilder sb = new StringBuilder();

        Tuple basic = counters.get(0);
        sb.append(basic.get(0));
        sb.append(":");
        for(int i = 1; i < counters.size(); i++) {
            Tuple current = counters.get(i);
            sb.append(current.get(0));
            double psi = calculatePsi(basic, current);
            sb.append(":");
            sb.append(psi);
        }

        Tuple output = TupleFactory.getInstance().newTuple(2);
        output.set(0, columnId);
        output.set(1, sb.toString());
        return output;
    }

    @SuppressWarnings("unchecked")
    private double calculatePsi(Tuple basic, Tuple current) throws ExecException {
        double psi = 0d;
        double basicTotal = (double) basic.get(2);
        double currentTotal = (double) current.get(2);

        List<Double> basicCounters = (List<Double>) basic.get(1);
        List<Double> currentCounters = (List<Double>) current.get(1);

        assert basicCounters.size() == currentCounters.size();

        for(int i = 0; i < basicCounters.size(); i++) {
            double basicRate = (basicCounters.get(i) / basicTotal);
            double currentRate = (currentCounters.get(i) / currentTotal);

            if(Double.compare(basicCounters.get(i), 0d) == 0 || Double.compare(currentCounters.get(i), 0d) == 0) {
                continue;
            }
            psi += ((currentRate - basicRate) * Math.log(currentRate / basicRate));
        }
        return psi;
    }

    /**
     * output the schema for evaluation score
     */
    public Schema outputSchema(Schema input) {
        try {
            Schema tupleSchema = new Schema();
            tupleSchema.add(new Schema.FieldSchema("columnId", DataType.INTEGER));
            tupleSchema.add(new Schema.FieldSchema("psi", DataType.CHARARRAY));
            return new Schema(new Schema.FieldSchema("PSIInfo", tupleSchema, DataType.TUPLE));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }
}
