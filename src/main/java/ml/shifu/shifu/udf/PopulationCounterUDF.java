package ml.shifu.shifu.udf;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.util.Utils;
import org.apache.pig.parser.ParserException;

import java.io.IOException;
import java.util.Iterator;
import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;
import ml.shifu.shifu.udf.stats.CategoryCounter;
import ml.shifu.shifu.udf.stats.Counter;
import ml.shifu.shifu.udf.stats.NumericCounter;

/**
 * Calculate the counter for each bin
 */
public class PopulationCounterUDF extends AbstractTrainerUDF<Tuple> {

    private Counter counter;

    public PopulationCounterUDF(String source, String pathModelConfig, String pathColumnConfig)
        throws IOException {
        super(source, pathModelConfig, pathColumnConfig);
    }

    @Override
    public Tuple exec(Tuple input) throws IOException {
        if (input == null || input.size() == 0) {
            return null;
        }

        Tuple groupInfo = (Tuple) input.get(0);
        DataBag bag = (DataBag) input.get(1);

        Integer columnId = (Integer) groupInfo.get(1);
        ColumnConfig columnConfig = columnConfigList.get(columnId);

        if (columnConfig.isCategorical()) {
            this.counter = new CategoryCounter(columnConfig.getBinCategory());
        } else if (columnConfig.isNumerical()){
            this.counter = new NumericCounter(columnConfig.getColumnName(), columnConfig.getBinBoundary());
        } else {
            return null;
        }

        Iterator<Tuple> iter = bag.iterator();
        while (iter.hasNext()) {
            Tuple tuple = iter.next();
            if (tuple != null && tuple.size() != 0) {
                Object value = tuple.get(2);
                counter.addData(value);
            }
        }

        List<Integer> dataBin = counter.getCounter();

        Tuple output = TupleFactory.getInstance().newTuple(2);
        output.set(0, columnId);
        output.set(1, StringUtils.join(dataBin, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));

        return output;
    }

    public Schema outputSchema(Schema input) {
        try {
            return Utils
                .getSchemaFromString("PopulationInfo:Tuple(columnId : int, population : chararray)");
        } catch (ParserException e) {
            log.debug("Error when generating output schema.", e);
            // just ignore
            return null;
        }
    }
}
