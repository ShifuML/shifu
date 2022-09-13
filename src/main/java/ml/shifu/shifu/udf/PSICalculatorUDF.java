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
package ml.shifu.shifu.udf;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

import org.apache.commons.lang.StringUtils;
import org.apache.commons.math3.stat.descriptive.SummaryStatistics;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.util.UDFContext;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.util.Constants;
import ml.shifu.shifu.util.Environment;

/**
 * Calculate the Population Stability Index
 */
public class PSICalculatorUDF extends AbstractTrainerUDF<Tuple> {

    /**
     * New PSI compute mode: if psi column defined, in each column by comparing category by category, like 20200101 to
     * 20200102 to get PSI trend of each variables.
     */
    private PSIByColumnUDF psiInstance;
    private boolean psiByCategory = false;

    public PSICalculatorUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

        if(UDFContext.getUDFContext() != null && UDFContext.getUDFContext().getJobConf() != null) {
            psiByCategory = Boolean.TRUE.toString().equalsIgnoreCase(
                    UDFContext.getUDFContext().getJobConf().get(Constants.SHIFU_PSI_BY_COLUMN_CATEGORY));
        } else {
            psiByCategory = Boolean.TRUE.toString()
                    .equalsIgnoreCase(Environment.getProperty(Constants.SHIFU_PSI_BY_COLUMN_CATEGORY));
        }

        if(this.psiByCategory) {
            psiInstance = new PSIByColumnUDF(source, pathModelConfig, pathColumnConfig);
        }
    }

    @Override
    public Tuple exec(Tuple input) throws IOException {
        if(input == null || input.size() < 2) {
            return null;
        }

        if(this.psiByCategory) {
            return this.psiInstance.exec(input);
        }

        Integer columnId = (Integer) input.get(0);
        DataBag databag = (DataBag) input.get(1);

        ColumnConfig columnConfig = this.columnConfigList.get(columnId);

        List<Integer> negativeBin = columnConfig.getBinCountNeg();
        List<Integer> positiveBin = columnConfig.getBinCountPos();
        List<Double> expected = new ArrayList<Double>(negativeBin.size());
        for(int i = 0; i < columnConfig.getBinCountNeg().size(); i++) {
            if(columnConfig.getTotalCount() == 0) {
                expected.add(0D);
            } else {
                expected.add(
                        ((double) negativeBin.get(i) + (double) positiveBin.get(i)) / columnConfig.getTotalCount());
            }
        }

        Double psi = 0D;
        double cosine = 0.0d;
        SummaryStatistics psiStats = new SummaryStatistics();
        SummaryStatistics cosStats = new SummaryStatistics();
        List<String> unitStats = new ArrayList<String>();

        Iterator<Tuple> iter = databag.iterator();
        while(iter.hasNext()) {
            Tuple tuple = iter.next();
            if(tuple != null && tuple.size() != 0) {
                double psiByPsiColumn = 0d;
                String subBinStr = (String) tuple.get(1);
                String[] subBinArr = StringUtils.split(subBinStr, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR);
                List<Double> subCounter = new ArrayList<Double>();
                Double total = 0D;
                for(String binningElement: subBinArr) {
                    Double dVal = Double.valueOf(binningElement);
                    subCounter.add(dVal);
                    total += dVal;
                }

                int i = 0;
                for (Double sub : subCounter) {
                    if (total != 0 && expected.get(i) != 0) {
                        double logNum = (sub / total) / expected.get(i);
                        if (logNum <= 0) {
                            i++;
                            continue;
                        } else {
                            double unitPsi = ((sub / total - expected.get(i)) * Math.log(logNum));
                            psi = psi + unitPsi;
                            psiByPsiColumn += unitPsi;
                            psiStats.addValue(unitPsi);
                        }
                    }
                    i++;
                }

                double unitCosine = calculateCosine(expected, subCounter);
                cosine = cosine + unitCosine;
                cosStats.addValue(unitCosine);
                unitStats.add((String) tuple.get(2) + "^" + psiByPsiColumn);
            }
        }

        // sort by unit
        Collections.sort(unitStats);

        Tuple output = TupleFactory.getInstance().newTuple(6);
        output.set(0, columnId);
        output.set(1, psi);
        output.set(2, psiStats.getStandardDeviation());
        output.set(3, cosine);
        output.set(4, cosStats.getStandardDeviation());
        output.set(5, StringUtils.join(unitStats, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));

        return output;
    }

    private double calculateCosine(List<Double> totalVector, List<Double> subVector) {
        assert totalVector != null && subVector != null && totalVector.size() != 0
                && totalVector.size() == subVector.size();

        double multi = 0.0d;
        double squareX = 0.0d, squareY = 0.0d;
        for(int i = 0; i < totalVector.size(); i++) {
            double x = totalVector.get(i);
            double y = subVector.get(i);

            multi = multi + x * y;
            squareX = squareX + x * x;
            squareY = squareY + y * y;
        }

        double divisor = Math.sqrt(squareX) * Math.sqrt(squareY);
        if(divisor < 1e-10) {
            return 0;
        }
        return multi / divisor;
    }

    /**
     * output the schema for evaluation score
     */
    public Schema outputSchema(Schema input) {
        try {
            Schema tupleSchema = new Schema();
            tupleSchema.add(new Schema.FieldSchema("columnId", DataType.INTEGER));
            if(this.psiByCategory) {
                tupleSchema.add(new Schema.FieldSchema("psi", DataType.CHARARRAY));
            } else {
                tupleSchema.add(new Schema.FieldSchema("psi", DataType.DOUBLE));
                tupleSchema.add(new Schema.FieldSchema("psiStd", DataType.DOUBLE));
                tupleSchema.add(new Schema.FieldSchema("cosine", DataType.DOUBLE));
                tupleSchema.add(new Schema.FieldSchema("cosStd", DataType.DOUBLE));
                tupleSchema.add(new Schema.FieldSchema("unitstats", DataType.CHARARRAY));
            }

            return new Schema(new Schema.FieldSchema("PSIInfo", tupleSchema, DataType.TUPLE));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }
}
