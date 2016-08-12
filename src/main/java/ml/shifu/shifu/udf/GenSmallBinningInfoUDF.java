package ml.shifu.shifu.udf;

import java.io.IOException;
import java.util.Iterator;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.core.binning.AbstractBinning;
import ml.shifu.shifu.core.binning.CategoricalBinning;
import ml.shifu.shifu.core.binning.EqualIntervalBinning;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;

/**
 * Created by zhanhu on 7/5/16.
 */
public class GenSmallBinningInfoUDF extends AbstractTrainerUDF<String> {

    private  int scaleFactor = 1000;

    public GenSmallBinningInfoUDF(String source, String pathModelConfig, String pathColumnConfig, String histoScaleFactor) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        this.scaleFactor = Integer.parseInt(histoScaleFactor);
    }

    @Override
    public String exec(Tuple input) throws IOException {
        if ( input == null || input.size() != 2 ) {
            return null;
        }

        Integer columnId = (Integer) input.get(0);
        DataBag dataBag = (DataBag) input.get(1);

        ColumnConfig columnConfig = super.columnConfigList.get(columnId);

        @SuppressWarnings("rawtypes")
        AbstractBinning binning = null;
        if ( columnConfig.isNumerical() ) {
            binning = new EqualIntervalBinning(this.scaleFactor, super.modelConfig.getMissingOrInvalidValues());
        } else {
            binning = new CategoricalBinning(this.scaleFactor, super.modelConfig.getMissingOrInvalidValues());
        }

        Iterator<Tuple> iterator = dataBag.iterator();
        while ( iterator.hasNext() ) {
            Tuple tuple = iterator.next();
            if ( tuple != null && tuple.size() >= 3 ) {
                String val = (String) tuple.get(1);
                binning.addData(val);
            }
        }

        return StringUtils.join(binning.getDataBin(), AbstractBinning.FIELD_SEPARATOR);
    }

}
