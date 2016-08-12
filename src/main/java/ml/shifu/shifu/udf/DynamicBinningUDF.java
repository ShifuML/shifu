package ml.shifu.shifu.udf;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.binning.AbstractBinning;
import ml.shifu.shifu.core.binning.DynamicBinning;
import ml.shifu.shifu.core.binning.obj.NumBinInfo;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.backend.executionengine.ExecException;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;

import java.io.IOException;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

/**
 * Created by zhanhu on 7/6/16.
 */
public class DynamicBinningUDF extends AbstractTrainerUDF<Tuple> {

    public DynamicBinningUDF(String source, String pathModelConfig, String pathColumnConfig) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
    }

    @Override
    public Tuple exec(Tuple input) throws IOException {

        if ( input == null || input.size() != 4 ) {
            return null;
        }

        Integer columnId = (Integer) input.get(0);
        DataBag columnDataBag = (DataBag) input.get(1);
        String binsData = (String) input.get(3);

        if ( columnDataBag == null || binsData == null ) {
            return null;
        }

        Tuple output = TupleFactory.getInstance().newTuple(2);
        ColumnConfig columnConfig = super.columnConfigList.get(columnId);

        output.set(0, columnId);
        if ( columnConfig.isCategorical() ) {
            output.set(1, binsData);
        } else {
            List<NumBinInfo> binInfoList = NumBinInfo.constructNumBinfo(binsData, AbstractBinning.FIELD_SEPARATOR);
            updateNumBinInfo(columnDataBag, binInfoList);

            DynamicBinning dynamicBinning = new DynamicBinning(binInfoList, modelConfig.getStats().getMaxNumBin());
            List<Double> binFields = dynamicBinning.getDataBin();
            output.set(1, StringUtils.join(binFields, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));
        }

        return output;
    }

    private void updateNumBinInfo(DataBag columnDataBag, List<NumBinInfo> binInfoList) throws ExecException {
        Set<String> missingValSet = new HashSet<String>(super.modelConfig.getMissingOrInvalidValues());

        Iterator<Tuple> iterator = columnDataBag.iterator();
        while ( iterator.hasNext() ) {
            Tuple tuple = iterator.next();
            String val = (String) tuple.get(1);
            Boolean isPositiveInst = (Boolean) tuple.get(2);

            if ( missingValSet.contains(val) ) {
                continue;
            }

            Double d = null;

            try {
                d = Double.valueOf(val);
            } catch ( Exception e ) {
                // illegal number, just skip it
                continue;
            }

            NumBinInfo numBinInfo = binaryLocate(binInfoList, d);
            if ( numBinInfo != null ) {
                numBinInfo.incInstCnt(isPositiveInst);
            }
        }
    }

    public NumBinInfo binaryLocate(List<NumBinInfo> binInfoList, Double d) {
        int left = 0;
        int right = binInfoList.size() - 1;

        while ( left <= right ) {
            int middle = (left + right) / 2;
            NumBinInfo binInfo = binInfoList.get(middle);
            if ( d >= binInfo.getLeftThreshold() && d < binInfo.getRightThreshold() ) {
                return binInfo;
            } else if ( d >= binInfo.getRightThreshold() ) {
                left = middle + 1;
            } else if ( d < binInfo.getLeftThreshold() ) {
                right = middle - 1;
            } else {
                return null;
            }
        }

        return null;
    }
}
