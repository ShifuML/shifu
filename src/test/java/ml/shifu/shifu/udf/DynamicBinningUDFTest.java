package ml.shifu.shifu.udf;

import java.io.IOException;
import java.io.InputStream;
import java.util.List;
import java.util.Random;
import java.util.zip.GZIPInputStream;

import ml.shifu.shifu.core.binning.DynamicBinningTest;
import ml.shifu.shifu.core.binning.EqualIntervalBinning;
import ml.shifu.shifu.core.binning.obj.NumBinInfo;

import org.apache.commons.io.IOUtils;
import org.apache.commons.lang.StringUtils;
import org.apache.pig.data.DataBag;
import org.apache.pig.data.DefaultDataBag;
import org.apache.pig.data.Tuple;
import org.apache.pig.data.TupleFactory;
import org.testng.Assert;

/**
 * Created by zhanhu on 7/6/16.
 */
public class DynamicBinningUDFTest {

//    @Test
    public void testDynamicBinningUDFTest() throws IOException {
        DynamicBinningUDF inst = new DynamicBinningUDF("LOCAL",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json",
                "src/test/resources/example/partfile");

        List<NumBinInfo> binInfoList = DynamicBinningTest.createNumBinInfos(10000);
        Random rd = new Random(System.currentTimeMillis());

        long startTs = System.currentTimeMillis();

        for (int i = 0; i < 10000; i++) {
            double val = rd.nextDouble() * 200;
            NumBinInfo numBinInfo = inst.binaryLocate(binInfoList, val);
            Assert.assertNotNull(numBinInfo);
        }

        System.out.println("Spend " + (System.currentTimeMillis() - startTs) + "-ms to query binning number according value.");
    }

//    @Test
    public void testDynamicBinningUDF2Test() throws IOException {
        DynamicBinningUDF inst = new DynamicBinningUDF("LOCAL",
                "src/test/resources/example/inner_seg1_v15/ModelConfig.json",
                "src/test/resources/example/inner_seg1_v15/ColumnConfig.json",
                "src/test/resources/example/inner_seg1_v15/smallbins");


        Tuple input = TupleFactory.getInstance().newTuple(1);
        input.set(0, createDataBag());

        String binsText = (String)inst.exec(input).get(1);
        Assert.assertEquals(StringUtils.split(binsText, CalculateStatsUDF.CATEGORY_VAL_SEPARATOR).length, 10);
    }

    private DataBag createDataBag() throws IOException {
        InputStream is = DynamicBinningUDFTest.class.getResourceAsStream("/example/inner_seg1_v15/dib_sample_fields.gz");
        GZIPInputStream gzis = new GZIPInputStream(is);

        DataBag databag = new DefaultDataBag();

        EqualIntervalBinning inst = new EqualIntervalBinning(1000);
        List<String> lines = IOUtils.readLines(gzis);
        for (String record : lines) {
            String[] fields = record.split("\\|");
            inst.addData(fields[0]);

            Tuple tuple = TupleFactory.getInstance().newTuple(4);
            tuple.set(0, 1);
            tuple.set(1, fields[0]);
            tuple.set(2, (fields[1].equals("1") ? Boolean.TRUE : Boolean.FALSE));
            tuple.set(3, 0);

            databag.add(tuple);
        }

        IOUtils.closeQuietly(gzis);
        IOUtils.closeQuietly(is);

        return databag;
    }

}
