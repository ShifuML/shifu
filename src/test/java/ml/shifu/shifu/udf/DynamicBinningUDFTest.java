package ml.shifu.shifu.udf;

import ml.shifu.shifu.core.binning.DynamicBinningTest;
import ml.shifu.shifu.core.binning.obj.NumBinInfo;
import org.testng.Assert;
import org.testng.annotations.Test;

import java.io.IOException;
import java.util.List;
import java.util.Random;

/**
 * Created by zhanhu on 7/6/16.
 */
public class DynamicBinningUDFTest {

    @Test
    public void testDynamicBinningUDFTest() throws IOException {

        DynamicBinningUDF inst = new DynamicBinningUDF("LOCAL",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ModelConfig.json",
                "src/test/resources/example/cancer-judgement/ModelStore/ModelSet1/ColumnConfig.json");

        List<NumBinInfo> binInfoList = DynamicBinningTest.createNumBinInfos(10000);
        Random rd = new Random(System.currentTimeMillis());

        long startTs = System.currentTimeMillis();

        for ( int i = 0; i < 10000; i ++ ) {
            double val = rd.nextDouble() * 200;
            NumBinInfo numBinInfo = inst.binaryLocate(binInfoList, val);
            Assert.assertNotNull(numBinInfo);
        }

        System.out.println("Spend " + (System.currentTimeMillis() - startTs) + "-ms to query binning number according value.");
    }
}
