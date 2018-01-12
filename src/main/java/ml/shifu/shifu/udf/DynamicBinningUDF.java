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

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.RawSourceData.SourceType;
import ml.shifu.shifu.core.binning.AbstractBinning;
import ml.shifu.shifu.core.binning.DynamicBinning;
import ml.shifu.shifu.core.binning.obj.NumBinInfo;
import ml.shifu.shifu.fs.ShifuFileUtils;
import org.apache.commons.collections.CollectionUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;

import java.io.IOException;
import java.util.*;

/**
 * Created by zhanhu on 7/6/16.
 */
public class DynamicBinningUDF extends AbstractTrainerUDF<Tuple> {

    private Map<Integer, String> smallBinsMap;

    public DynamicBinningUDF(String source, String pathModelConfig, String pathColumnConfig, String smallBinsPath)
            throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

        smallBinsMap = new HashMap<Integer, String>();
        List<String> smallBinsList = ShifuFileUtils.readFilePartsIntoList(smallBinsPath, SourceType.HDFS);
        for(String smallBin: smallBinsList) {
            String[] fields = StringUtils.split(smallBin, '\u0007');
            if(fields.length == 2) {
                smallBinsMap.put(Integer.parseInt(fields[0]), fields[1]);
            }
        }
    }

    @Override
    public Tuple exec(Tuple input) throws IOException {

        if(input == null || input.size() != 1) {
            return null;
        }

        Integer columnId = null;
        ColumnConfig columnConfig = null;
        String binsData = null;

        Set<String> missingValSet = new HashSet<String>(super.modelConfig.getMissingOrInvalidValues());
        List<NumBinInfo> binInfoList = null;

        DataBag columnDataBag = (DataBag) input.get(0);
        Iterator<Tuple> iterator = columnDataBag.iterator();
        while(iterator.hasNext()) {
            Tuple tuple = iterator.next();
            if(columnId == null) {
                columnId = (Integer) tuple.get(0);

                // for filter expansions
                if(columnId >= super.columnConfigList.size()) {
                    int newColumnId = columnId % super.columnConfigList.size();
                    columnConfig = super.columnConfigList.get(newColumnId);
                } else {
                    columnConfig = super.columnConfigList.get(columnId);
                }

                String smallBins = smallBinsMap.get(columnId);
                if(columnConfig.isCategorical()) {
                    binsData = smallBins;
                    break;
                } else {
                    binInfoList = NumBinInfo.constructNumBinfo(smallBins, AbstractBinning.FIELD_SEPARATOR);
                }
            }

            String val = (String) tuple.get(1);
            Boolean isPositiveInst = (Boolean) tuple.get(2);

            if(missingValSet.contains(val)) {
                continue;
            }

            Double d = null;

            try {
                d = Double.valueOf(val);
            } catch (Exception e) {
                // illegal number, just skip it
                continue;
            }

            NumBinInfo numBinInfo = binaryLocate(binInfoList, d);
            if(numBinInfo != null) {
                numBinInfo.incInstCnt(isPositiveInst);
            }
        }

        if(binsData == null && CollectionUtils.isNotEmpty(binInfoList)) {
            int maxNumBin = modelConfig.getStats().getMaxNumBin();
            if(maxNumBin <= 0) {
                maxNumBin = 1024;
            }
            DynamicBinning dynamicBinning = new DynamicBinning(binInfoList, maxNumBin);
            List<Double> binFields = dynamicBinning.getDataBin();
            binsData = StringUtils.join(binFields, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR);
        }

        Tuple output = TupleFactory.getInstance().newTuple(2);

        output.set(0, columnId);
        output.set(1, binsData);

        return output;
    }

    public NumBinInfo binaryLocate(List<NumBinInfo> binInfoList, Double d) {
        int left = 0;
        int right = binInfoList.size() - 1;

        while(left <= right) {
            int middle = (left + right) / 2;
            NumBinInfo binInfo = binInfoList.get(middle);
            if(d >= binInfo.getLeftThreshold() && d < binInfo.getRightThreshold()) {
                return binInfo;
            } else if(d >= binInfo.getRightThreshold()) {
                left = middle + 1;
            } else if(d < binInfo.getLeftThreshold()) {
                right = middle - 1;
            } else {
                return null;
            }
        }

        return null;
    }
}
