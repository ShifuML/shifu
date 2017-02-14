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

import ml.shifu.shifu.container.ValueObject;
import ml.shifu.shifu.core.BasicStatsCalculator;
import ml.shifu.shifu.core.Binning;
import ml.shifu.shifu.core.Binning.BinningDataType;
import ml.shifu.shifu.core.ColumnStatsCalculator;
import ml.shifu.shifu.core.ColumnStatsCalculator.ColumnMetrics;

import ml.shifu.shifu.util.CommonUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;

import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * CalculateStatsUDF class is calculate the stats for each column
 */
public class CalculateStatsUDF extends AbstractTrainerUDF<Tuple> {

    public static final char CATEGORY_VAL_SEPARATOR = '\u0001';

    private Double valueThreshold = 1e6;

    DecimalFormat df = new DecimalFormat("##.######");

    public CalculateStatsUDF(String source, String pathModelConfig, String pathColumnConfig, String withScoreStr)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

        if(modelConfig.getNumericalValueThreshold() != null) {
            valueThreshold = modelConfig.getNumericalValueThreshold();
        }
        log.debug("Value Threshold: " + valueThreshold);
    }

    public Tuple exec(Tuple input) throws IOException {
        if(input == null || input.size() == 0) {
            return null;
        }

        TupleFactory tupleFactory = TupleFactory.getInstance();

        Integer columnNum = (Integer) input.get(0);
        DataBag bag = (DataBag) input.get(1);

        BinningDataType dataType;
        if(modelConfig.isCategoricalDisabled()) {
            dataType = BinningDataType.Numerical;
        } else {
            if(columnConfigList.get(columnNum).isCategorical()) {
                dataType = BinningDataType.Categorical;
            } else if(columnConfigList.get(columnNum).isNumerical()) {
                dataType = BinningDataType.Numerical;
            } else if(modelConfig.isBinningAutoTypeEnabled()) {
                // if type is Auto, and the auto type enable is true
                dataType = BinningDataType.Auto;
            } else {
                // if type is Auto, but the auto type enable is false
                dataType = BinningDataType.Numerical;
            }
        }

        List<ValueObject> voList = new ArrayList<ValueObject>();
        Iterator<Tuple> iterator = bag.iterator();
        log.debug("****** The element count in bag is : " + bag.size());

        long total = 0l;
        long missing = 0l;

        while(iterator.hasNext()) {

            total++;

            Tuple t = iterator.next();
            if(t.get(1) == null) {
                missing++;
                continue;
            }

            ValueObject vo = new ValueObject();
            String valueStr = ((t.get(0) == null) ? "" : t.get(0).toString());

            if(dataType.equals(BinningDataType.Numerical)) {
                Double value = null;
                try {
                    value = Double.valueOf(valueStr);
                } catch (NumberFormatException e) {
                    // if there are too many log, it will case ReduceTask - `java.lang.OutOfMemoryError: Java heap
                    // space`
                    // log.warn("Incorrect data, not numerical - " + valueStr);
                    missing++;
                    continue;
                }

                if(value > valueThreshold) {
                    log.warn("Exceed Threshold: " + value + " / " + valueThreshold);
                    missing++;
                    continue;
                }

                vo.setValue(value);
            } else {
                // Categorical or Auto
                if(StringUtils.isEmpty(valueStr)) {
                    missing++;
                }
                vo.setRaw(valueStr);
            }
            // do not need to catch exception, see AddColumnNumUDF which have already normalized the weight value
            vo.setWeight(Double.valueOf(t.get(2).toString()));

            vo.setTag(CommonUtils.trimTag(t.get(1).toString()));
            // vo.setScore(Double.valueOf(t.get(2).toString()));
            voList.add(vo);
        }

        if(voList.size() < 10) {
            return null;
        }

        // Calculate Binning
        Binning binning = new Binning(modelConfig.getPosTags(), modelConfig.getNegTags(), dataType, voList);
        binning.setMaxNumOfBins(modelConfig.getBinningExpectedNum());
        binning.setBinningMethod(modelConfig.getBinningMethod());
        binning.setAutoTypeThreshold(modelConfig.getBinningAutoTypeThreshold());
        binning.setMergeEnabled(modelConfig.isBinningMergeEnabled());
        binning.doBinning();

        // Calculate Basic Stats
        BasicStatsCalculator basicStatsCalculator = new BasicStatsCalculator(binning.getUpdatedVoList(),
                this.valueThreshold);

        ColumnMetrics columnCountMetrics = ColumnStatsCalculator.calculateColumnMetrics(binning.getBinCountNeg(),
                binning.getBinCountPos());

        // Assemble the results
        Tuple tuple = tupleFactory.newTuple();
        tuple.append(columnNum);
        if(binning.getUpdatedDataType().equals(BinningDataType.Categorical)) {
            tuple.append("[" + StringUtils.join(binning.getBinCategory(), CATEGORY_VAL_SEPARATOR) + "]");
        } else {
            tuple.append(binning.getBinBoundary().toString());
        }
        tuple.append(binning.getBinCountNeg().toString());
        tuple.append(binning.getBinCountPos().toString());
        // tuple.append(null);
        tuple.append(binning.getBinAvgScore().toString());
        tuple.append(binning.getBinPosCaseRate().toString());
        tuple.append(df.format(columnCountMetrics.getKs()));
        tuple.append(df.format(columnCountMetrics.getIv()));

        tuple.append(df.format(basicStatsCalculator.getMax()));
        tuple.append(df.format(basicStatsCalculator.getMin()));
        tuple.append(df.format(basicStatsCalculator.getMean()));
        tuple.append(df.format(basicStatsCalculator.getStdDev()));

        if(binning.getUpdatedDataType().equals(BinningDataType.Numerical)) {
            tuple.append("N");
        } else {
            tuple.append("C");
        }

        tuple.append(df.format(basicStatsCalculator.getMedian()));
        tuple.append(df.format(missing));
        tuple.append(df.format(total));
        tuple.append(df.format((double) missing / total));
        tuple.append(binning.getBinWeightedNeg().toString());
        tuple.append(binning.getBinWeightedPos().toString());

        return tuple;

    }

}
