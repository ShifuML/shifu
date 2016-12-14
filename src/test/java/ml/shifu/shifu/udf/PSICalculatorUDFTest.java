package ml.shifu.shifu.udf;

import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.DefaultDataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;

/**
 * Created by Mark on 5/31/2016.
 */
public class PSICalculatorUDFTest {

    private PSICalculatorUDF inst;
    private Integer[] array;

    @BeforeClass
    public void setup() throws IOException {
        inst = new PSICalculatorUDF("LOCAL",
                                        "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                                        "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json");

        array = new Integer[] {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    }


    @Test
    public void testCase1() throws IOException {
        Tuple input = TupleFactory.getInstance().newTuple(2);

        input.set(0, Integer.valueOf(1));

        DataBag dataBag = new DefaultDataBag();
        Tuple tuple = TupleFactory.getInstance().newTuple(4);
        tuple.set(0, Integer.valueOf(1));
        tuple.set(1, StringUtils.join(array, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));
        tuple.set(2, "2015/06/20");
        tuple.set(3, 3.14d);
        dataBag.add(tuple);

        input.set(1, dataBag);

        System.out.println(inst.exec(input));
    }


}
