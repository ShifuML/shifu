package ml.shifu.shifu.udf;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.container.obj.ModelStatsConf;
import ml.shifu.shifu.core.binning.AbstractBinning;
import ml.shifu.shifu.core.binning.CategoricalBinning;
import ml.shifu.shifu.core.binning.EqualIntervalBinning;
import ml.shifu.shifu.core.binning.MunroPatBinning;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.DataType;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;

import java.io.IOException;
import java.util.Iterator;

/**
 * Created by zhanhu on 7/5/16.
 */
public class GenSmallBinningInfoUDF extends AbstractTrainerUDF<Tuple> {

    private  int scaleFactor = 1000;

    public GenSmallBinningInfoUDF(String source, String pathModelConfig, String pathColumnConfig, String histoScaleFactor) throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
        this.scaleFactor = Integer.parseInt(histoScaleFactor);
    }

    @Override
    public Tuple exec(Tuple input) throws IOException {
        if ( input == null || input.size() != 1 ) {
            return null;
        }

        Integer columnId = null;
        ColumnConfig columnConfig = null;
        AbstractBinning binning = null;

        DataBag dataBag = (DataBag) input.get(0);
        Iterator<Tuple> iterator = dataBag.iterator();
        while ( iterator.hasNext() ) {
            Tuple tuple = iterator.next();
            if ( tuple != null && tuple.size() >= 3 ) {
                if ( columnId == null ) {
                    columnId = (Integer) tuple.get(0);
                    columnConfig = super.columnConfigList.get(columnId);
                    binning = getBinningHandler(columnConfig);
                }

                Boolean isPostive = (Boolean)tuple.get(2);
                if ( isToBinningVal(columnConfig, isPostive) ) {
                    String val = (String) tuple.get(1);
                    binning.addData(val);
                }
            }
        }

        Tuple output = TupleFactory.getInstance().newTuple(2);
        output.set(0, columnId);
        output.set(1, StringUtils.join(binning.getDataBin(), AbstractBinning.FIELD_SEPARATOR));

        return output;
    }

    private boolean isToBinningVal(ColumnConfig columnConfig, Boolean isPostive) {
        return columnConfig.isCategorical()
                || super.modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.EqualTotal)
                || super.modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.EqualInterval)
                || (super.modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.EqualPositive) && isPostive)
                || (super.modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.EqualNegtive) && !isPostive );
    }

    private AbstractBinning getBinningHandler(ColumnConfig columnConfig) {
        AbstractBinning binning = null;
        if ( columnConfig.isNumerical() ) {
            if ( modelConfig.getBinningMethod().equals(ModelStatsConf.BinningMethod.EqualInterval) ) {
                binning = new EqualIntervalBinning(this.scaleFactor, super.modelConfig.getMissingOrInvalidValues());
            } else {
                binning = new MunroPatBinning(this.scaleFactor, super.modelConfig.getMissingOrInvalidValues());
            }
        } else {
            binning = new CategoricalBinning(this.scaleFactor, super.modelConfig.getMissingOrInvalidValues());
        }

        return binning;
    }

    public Schema outputSchema(Schema input) {
        try {
            Schema tupleSchema = new Schema();
            tupleSchema.add(new Schema.FieldSchema("columnId", DataType.INTEGER));
            tupleSchema.add(new Schema.FieldSchema("bins", DataType.CHARARRAY));

            return new Schema(new Schema.FieldSchema("binning", tupleSchema, DataType.TUPLE));
        } catch (IOException e) {
            log.error("Error in outputSchema", e);
            return null;
        }
    }

}
