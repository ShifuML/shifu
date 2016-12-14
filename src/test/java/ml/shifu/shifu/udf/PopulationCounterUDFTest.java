package ml.shifu.shifu.udf;

import org.apache.pig.data.DataBag;
import org.apache.pig.data.DefaultDataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

import java.io.IOException;

/**
 * Created by Mark on 5/30/2016.
 */
public class PopulationCounterUDFTest {

    private PopulationCounterUDF inst;
    private double[] array;


    @BeforeClass
    public void setup() throws IOException {
        inst = new PopulationCounterUDF("LOCAL",
                                        "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                                        "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json",
                                        "2");

        array = new double[] {13D, 14D, 14D, 17D, 17D, 18D, 19D, 20D, 21D, 27D};
    }




    @Test
    public void testCase1() throws IOException {

        Tuple input = TupleFactory.getInstance().newTuple(2);
        Tuple groupInfo = TupleFactory.getInstance().newTuple(2);

        groupInfo.set(0, "column_3");
        groupInfo.set(1, Integer.valueOf(1));

        DataBag dataBag = new DefaultDataBag();
        //{(PSIColumn: int, columnId: int, value: chararray, tag: boolean , rand: int)}
        for (int i = 0 ; i < 10; i ++) {
            Tuple t = TupleFactory.getInstance().newTuple(4);
            t.set(0, Integer.valueOf(1));
            t.set(1, Integer.valueOf(1));
            t.set(2, array[i]);
            dataBag.add(t);
        }

        input.set(0, groupInfo);
        input.set(1, dataBag);

        Tuple output = inst.exec(input);

        Assert.assertEquals(output.get(0), 1);

        String[] outputArray = output.get(1).toString().split(String.valueOf(CalculateStatsUDF.CATEGORY_VAL_SEPARATOR));

        Assert.assertEquals(outputArray[0], "1");
        Assert.assertEquals(outputArray[1], "2");
        Assert.assertEquals(outputArray[2], "0");
    }


}
