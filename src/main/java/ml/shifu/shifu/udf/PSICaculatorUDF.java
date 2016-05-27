package ml.shifu.shifu.udf;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.apache.pig.impl.logicalLayer.schema.Schema;
import org.apache.pig.impl.util.Utils;
import org.apache.pig.parser.ParserException;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import ml.shifu.shifu.container.obj.ColumnConfig;

/**
 * Calculate the Population Stability Index
 */
public class PSICaculatorUDF extends AbstractTrainerUDF<Tuple> {

    public PSICaculatorUDF(String source, String pathModelConfig, String pathColumnConfig)
        throws IOException {
        super(source, pathModelConfig, pathColumnConfig);

    }

    @Override
    public Tuple exec(Tuple input) throws IOException {
        if(input == null || input.size() < 2) {
            return null;
        }

        Integer columnId = (Integer) input.get(0);
        DataBag databag = (DataBag) input.get(1);

        ColumnConfig columnConfig = this.columnConfigList.get(columnId);

        List<Integer> negativeBin = columnConfig.getBinCountNeg();
        List<Integer> postiveBin  = columnConfig.getBinCountPos();
        List<Double> expected = new ArrayList<Double>(negativeBin.size());
        for (int i = 0 ; i < columnConfig.getBinCountNeg().size(); i ++) {
            expected.set(i, (double)(negativeBin.get(i) + postiveBin.get(i)) / columnConfig.getTotalCount());
        }

        Iterator<Tuple> iter = databag.iterator();
        Double psi = 0D;

        while (iter.hasNext()) {
            Tuple tuple = iter.next();
            if (tuple != null && tuple.size() != 0) {
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
                    // TODO Is it possible total == 0? or expect == actual ?
                    psi = psi + ((sub / total - expected.get(i)) * Math.log(sub / total - expected.get(i)));
                    i ++;
                }
            }
        }

        Tuple output = TupleFactory.getInstance().newTuple(2);
        output.set(0, columnId);
        output.set(1, psi);
        return output;
    }

    public Schema outputSchema(Schema input) {
        try {
            return Utils
                .getSchemaFromString("PSIInfo:Tuple(columnId : int, psi : double)");
        } catch (ParserException e) {
            log.debug("Error when generating output schema.", e);
            // just ignore
            return null;
        }
    }
}
